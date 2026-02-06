#!/usr/bin/env python3
"""
Multi-run ROC plotting with:
  - ROC curve from the fold with the HIGHEST TPR@0.1%FPR (using OOF predictions or Shadow OOF if provided)
  - Ability to DROP metrics by original name
  - Ability to RENAME metrics via old=new pairs
  - Column-wise subplot filling order (top->bottom, then next column)
  - Shadow model training (--shadow-npz) with ID-based alignment (no need for equal counts)

Notes:
- Drops are matched against ORIGINAL metric names (from CONF.base_feature_names plus
  'avg_base_score' and 'avg_ft_score').
- Renames only affect display names (legends/titles), not data access.
- Shadow mode: we align target and shadow prompts by a stable key (auto-detected or via --align-key),
  train each fold on SHADOW(train) and predict on TARGET(test).
- NEW (per your request): when shadow is used, the per-metric dashed curves use metrics from the
  SHADOW side (aligned) rather than TARGET.
- NEW (per your request): The main ROC curve now represents the fold with the highest TPR@0.1%FPR.
"""

import argparse
from pathlib import Path
import math
from typing import Optional, Tuple, List, Dict, Iterable
import warnings

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

# Local imports
from src.utils import load_prompt_data_from_npz, PromptData
from src.config import CONF

# Optional aesthetics (safe to leave)
import seaborn as sns  # noqa: F401


# -----------------------------
# CLI helpers
# -----------------------------
def parse_renames(pairs: Optional[List[str]]) -> Dict[str, str]:
    """Parse --rename 'old=new' items into a dict."""
    mapping = {}
    for item in pairs or []:
        if "=" not in item:
            raise ValueError(f"--rename expects 'old=new' format, got: {item}")
        old, new = item.split("=", 1)
        mapping[old.strip()] = new.strip()
    return mapping


# -----------------------------
# Metric extraction
# -----------------------------
def _metric_specs(rename_map: dict, drop_set: set):
    """
    Build a list of (orig_name, plot_name, getter) for all metrics,
    honoring drop_set and rename_map.
    """
    specs = []

    # Base features from CONF
    for n in CONF.base_feature_names:
        if n in drop_set:
            continue
        plot_name = rename_map.get(n, n)
        specs.append((n, plot_name, lambda p, n=n: float(p.features[n])))
    # Aggregated scores appended
    for n, getter in (
        ("avg_base_score", lambda p: float(p.avg_base_score)),
        ("avg_ft_score",   lambda p: float(p.avg_ft_score)),
    ):
        if n in drop_set:
            continue
        plot_name = rename_map.get(n, n)
        specs.append((n, plot_name, getter))
    print(specs)
    return specs


def _build_matrix_and_labels(
    prompts: List[PromptData],
    specs,
    return_keys: bool = False,
    key_attr: Optional[str] = None
) -> Tuple[np.ndarray, np.ndarray, List[str], Optional[List]]:
    """Return X (N x D), y (N,), plot_names (D-list), and optional keys (N-list)."""
    X = []
    y = []
    keys = [] if return_keys else None
    for p in prompts:
        vec = [getter(p) for (_, _, getter) in specs]
        X.append(vec)
        y.append(p.label)
        if return_keys:
            keys.append(_get_prompt_key(p, preferred=key_attr))
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=int)
    plot_names = [plot_name for (_, plot_name, _) in specs]
    return X, y, plot_names, keys


def clean_infs(X: np.ndarray) -> np.ndarray:
    X = X.astype(float, copy=True)
    inf_mask = ~np.isfinite(X)
    if inf_mask.any():
        X[inf_mask] = np.nan
    return X


# -----------------------------
# ID/key alignment
# -----------------------------
CANDIDATE_KEY_ATTRS = ("prompt_id", "uid", "id", "example_id")

def _detect_key_attr(prompts: List[PromptData]) -> Optional[str]:
    """Try to auto-detect a stable key attribute present on all prompts."""
    if not prompts:
        return None
    for attr in CANDIDATE_KEY_ATTRS:
        ok = True
        for p in prompts:
            if not hasattr(p, attr):
                ok = False
                break
            if getattr(p, attr) is None:
                ok = False
                break
        if ok:
            return attr
    return None


def _get_prompt_key(p: PromptData, preferred: Optional[str] = None):
    """Extract a stable key for a prompt. Prefer 'preferred'; otherwise try candidates; else None."""
    if preferred and hasattr(p, preferred):
        return getattr(p, preferred)
    for attr in CANDIDATE_KEY_ATTRS:
        if hasattr(p, attr):
            val = getattr(p, attr)
            if val is not None:
                return val
    return None  # Will trigger index-based fallback if necessary


def _index_by_key(keys: Iterable) -> Dict:
    return {k: i for i, k in enumerate(keys)}


def _align_by_keys(
    X_t: np.ndarray, y_t: np.ndarray, keys_t: Optional[List],
    X_s: np.ndarray, y_s: np.ndarray, keys_s: Optional[List],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Align target and shadow matrices by intersection of keys.
    If keys are missing, fallback to index-based min length with a warning.

    Returns:
      X_t_aln, y_t_aln, X_s_aln, y_s_aln, keep_mask (indices into the *aligned* arrays)
    """
    if keys_t is None or keys_s is None or any(k is None for k in keys_t) or any(k is None for k in keys_s):
        # Fallback: index-based alignment
        n = min(len(y_t), len(y_s))
        if (len(y_t) != len(y_s)) or (n == 0):
            print("[WARN] No alignable keys found; falling back to index-based alignment. "
                  f"Using first {n} rows of each. Consider adding --align-key.")
        return X_t[:n], y_t[:n], X_s[:n], y_s[:n], np.arange(n, dtype=int)

    # Key-based alignment
    idx_s = _index_by_key(keys_s)
    common_keys = [k for k in keys_t if k in idx_s]
    if not common_keys:
        raise ValueError("No overlapping keys between target and shadow NPZs. "
                         "Provide a correct --align-key or ensure IDs overlap.")

    # Build aligned arrays in the order of TARGET keys (filtered to intersection)
    t_indices = []
    s_indices = []
    for i_t, k in enumerate(keys_t):
        if k in idx_s:
            t_indices.append(i_t)
            s_indices.append(idx_s[k])

    X_t_aln = X_t[np.array(t_indices)]
    y_t_aln = y_t[np.array(t_indices)]
    X_s_aln = X_s[np.array(s_indices)]
    y_s_aln = y_s[np.array(s_indices)]
    keep = np.arange(len(t_indices), dtype=int)

    dropped_t = len(y_t) - len(t_indices)
    dropped_s = len(y_s) - len(t_indices)
    print(f"[ALIGN] Common={len(t_indices)} | dropped target-only={dropped_t} | dropped shadow-only={dropped_s}")
    return X_t_aln, y_t_aln, X_s_aln, y_s_aln, keep


# -----------------------------
# Model builders & OOF scoring (Modified for Best Fold)
# -----------------------------
def _build_stacker(random_state: int):
    """Factory to build the stacker so it's refit each fold (train-only preprocessing)."""
    return StackingClassifier(
        estimators=[
            ('rf', RandomForestClassifier(n_estimators=100, random_state=CONF.random_state)),
            ('lr', make_pipeline(SimpleImputer(strategy="median", add_indicator=True),
                                 StandardScaler(with_mean=False),
                                 LogisticRegression(C=0.5,random_state=CONF.random_state)))
        ],
        final_estimator=LogisticRegression(),
        cv=5
    )

def _tpr_at_fpr(y_true, y_scores, target_fpr=0.001):
    """Calculate TPR at the highest achievable TPR where FPR <= target_fpr."""
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    # Find indices where FPR <= target_fpr
    valid_indices = fpr <= target_fpr
    if not np.any(valid_indices):
        # If no point meets the FPR constraint, return 0 or find the closest
        # Find the index with the closest FPR to target_fpr
        closest_idx = np.argmin(np.abs(fpr - target_fpr))
        return tpr[closest_idx]
    # Get the maximum TPR among valid indices
    max_tpr_idx = np.argmax(tpr[valid_indices])
    # Map back to original index to get the TPR value
    original_valid_indices = np.where(valid_indices)[0]
    return tpr[original_valid_indices[max_tpr_idx]]

def _find_best_fold_scores(X, y, n_splits=5, random_state=2025, target_fpr=0.001):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    all_oof_scores = np.zeros(len(y), dtype=float)
    fold_tpr_at_fpr_list = []
    fold_auc_list = []
    fold_indices_list = [] # Stores (train_idx, test_idx) tuples

    for fold_num, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]

        X_tr = clean_infs(X_tr)
        X_te = clean_infs(X_te)

        stack = _build_stacker(random_state)
        stack.fit(X_tr, y_tr)
        fold_oof_scores = stack.predict_proba(X_te)[:, 1]
        all_oof_scores[test_idx] = fold_oof_scores

        # TPR@FPR
        fold_tpr_at_fpr = _tpr_at_fpr(y_te, fold_oof_scores, target_fpr=target_fpr)
        fold_tpr_at_fpr_list.append(fold_tpr_at_fpr)

        # AUC for this fold
        fold_auc = auc(*roc_curve(y_te, fold_oof_scores)[:2])
        fold_auc_list.append(fold_auc)

        # SAVE BOTH TRAIN AND TEST INDICES
        fold_indices_list.append((train_idx, test_idx))
        print(f"  Fold {fold_num}: TPR@{target_fpr*100}%FPR = {fold_tpr_at_fpr:.4f}, AUC = {fold_auc:.4f}")

    # Median fold by TPR@FPR
    sorted_indices = np.argsort(fold_tpr_at_fpr_list)
    median_idx = sorted_indices[len(sorted_indices)//2+1] # Max TPR fold (as per original logic logic "highest TPR") or median depending on your preference. 
    # NOTE: Original code comment said "Median fold" but logic selected max index if sorted ascending? 
    # If you want strictly BEST TPR:
    # best_idx = np.argmax(fold_tpr_at_fpr_list)
    best_idx = median_idx
    best_fold_train_indices, best_fold_test_indices = fold_indices_list[best_idx]
    best_fold_scores = all_oof_scores[best_fold_test_indices]
    best_fold_tpr = fold_tpr_at_fpr_list[best_idx]

    mean_auc = np.mean(fold_auc_list)
    print(f"  Selected fold (idx={best_idx}) TPR={best_fold_tpr:.4f}")
    print(f"  Mean AUC over {n_splits} folds: {mean_auc:.4f}")

    return best_fold_scores, best_fold_train_indices, best_fold_test_indices, best_fold_tpr, mean_auc

def _find_best_fold_scores_shadow(
    X_target: np.ndarray,
    y_target: np.ndarray,
    X_shadow: np.ndarray,
    y_shadow: np.ndarray,
    n_splits: int = 5,
    random_state: int = 2025,
    target_fpr: float = 0.001
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    
    # ... checks ...

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    all_target_oof_scores = np.zeros(len(y_target), dtype=float)
    fold_tpr_at_fpr_list = []
    fold_auc_list = []
    fold_indices_list = []

    for fold_num, (train_idx, test_idx) in enumerate(skf.split(X_target, y_target)):
        # Train on SHADOW(train), Predict on TARGET(test)
        X_tr_shadow, y_tr_shadow = X_shadow[train_idx], y_shadow[train_idx]
        X_te_target, y_te_target = X_target[test_idx], y_target[test_idx]

        X_tr_shadow = clean_infs(X_tr_shadow)
        X_te_target = clean_infs(X_te_target)

        stack = _build_stacker(random_state)
        stack.fit(X_tr_shadow, y_tr_shadow)
        fold_target_oof_scores = stack.predict_proba(X_te_target)[:, 1]
        all_target_oof_scores[test_idx] = fold_target_oof_scores

        fold_tpr = _tpr_at_fpr(y_te_target, fold_target_oof_scores, target_fpr=target_fpr)
        fold_tpr_at_fpr_list.append(fold_tpr)

        fold_auc = auc(*roc_curve(y_te_target, fold_target_oof_scores)[:2])
        fold_auc_list.append(fold_auc)

        fold_indices_list.append((train_idx, test_idx))
        print(f"  Fold {fold_num}: TPR@{target_fpr*100}%FPR (Target) = {fold_tpr:.4f}, AUC = {fold_auc:.4f}")

    best_idx = np.argmax(fold_tpr_at_fpr_list)
    best_fold_train_indices, best_fold_test_indices = fold_indices_list[best_idx]
    best_fold_scores = all_target_oof_scores[best_fold_test_indices]
    best_fold_tpr = fold_tpr_at_fpr_list[best_idx]
    mean_auc = np.mean(fold_auc_list)

    return best_fold_scores, best_fold_train_indices, best_fold_test_indices, best_fold_tpr, mean_auc

# -----------------------------
# Plotting
# -----------------------------
def _calibrate_and_predict(X_train_col, y_train, X_test_col):
    """
    Fits a simple LogisticRegression on a single feature column.
    Handles Infs/NaNs explicitly to prevent crashes.
    """
    # 1. 辅助函数：将 Inf 转为 NaN，以便 Imputer 处理
    def _safe_clean(arr):
        arr = arr.astype(float, copy=True)
        # 将 inf/-inf 标记为 NaN
        mask_inf = ~np.isfinite(arr) 
        if mask_inf.any():
            arr[mask_inf] = np.nan
        return arr.reshape(-1, 1)

    # 2. 清理数据
    X_tr = _safe_clean(X_train_col)
    X_te = _safe_clean(X_test_col)
    
    pipe = make_pipeline(
        # SimpleImputer 会把上面转出来的 NaN 填补为中位数
        SimpleImputer(strategy="median", add_indicator=True),
        StandardScaler(),
        LogisticRegression(C=0.5, solver='liblinear', random_state=CONF.random_state)
    )
    
    try:
        pipe.fit(X_tr, y_train)
        probs = pipe.predict_proba(X_te)[:, 1]
        return probs
    except Exception as e:
        print(f"[WARN] Failed to calibrate single feature: {e}. Falling back to cleaned raw values.")
        
        # 3. Fallback 逻辑修正：
        # 即使回退到原始值，也必须把 Inf 去掉，否则 roc_curve 会崩
        fallback = X_test_col.astype(float, copy=True)
        mask = ~np.isfinite(fallback)
        if mask.any():
            # 如果原始值有 inf/nan，用最小值或0填补，保证 roc_curve 能运行
            finite_vals = fallback[~mask]
            fill_val = np.min(finite_vals) if finite_vals.size > 0 else 0.0
            fallback[mask] = fill_val
        return fallback
    
def _plot_single_axes(
    ax,
    X, y, plot_names, title, random_state,
    X_shadow: Optional[np.ndarray] = None,
    y_shadow: Optional[np.ndarray] = None
):
    print(f"[{title}] Plotting...")
    ax.set_facecolor('#EAEAF2')
    ax.grid(True, color='white', linewidth=1)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    for spine in ax.spines.values():
        spine.set_edgecolor('#CCCCCC')

    target_fpr = 0.001
    
    # --- 1. Get Best Fold Data (Scores + Indices) ---
    if X_shadow is not None and y_shadow is not None:
        # Shadow Mode
        unified_scores, train_idx, test_idx, best_fold_tpr, mean_auc = _find_best_fold_scores_shadow(
            X_target=X, y_target=y, X_shadow=X_shadow, y_shadow=y_shadow,
            n_splits=5, random_state=random_state, target_fpr=target_fpr
        )
        # Training data comes from Shadow, Test data comes from Target
        X_train_for_metrics = X_shadow[train_idx]
        y_train_for_metrics = y_shadow[train_idx]
        X_test_for_metrics = X[test_idx]
        y_test_for_metrics = y[test_idx]
        metrics_src_tag = " (Shadow-Calibrated)"
    else:
        # Standard Mode
        unified_scores, train_idx, test_idx, best_fold_tpr, mean_auc = _find_best_fold_scores(
            X, y, n_splits=5, random_state=random_state, target_fpr=target_fpr
        )
        # Training and Test data come from Target
        X_train_for_metrics = X[train_idx]
        y_train_for_metrics = y[train_idx]
        X_test_for_metrics = X[test_idx]
        y_test_for_metrics = y[test_idx]
        metrics_src_tag = ""

    # y_test_for_metrics is the ground truth for the plotted curves
    
    # --- 2. Plot Unified (Stacker) ROC ---
    fpr_u, tpr_u, _ = roc_curve(y_test_for_metrics, unified_scores)
    legend_name = f"DIBA (AUC={mean_auc:.3f})"
    ax.plot(fpr_u, tpr_u, label=legend_name, linewidth=2, zorder=10)

    # --- 3. Plot Individual Metrics (Calibrated) ---
    # Now we iterate features, train on train_idx, predict on test_idx
    for j, name in enumerate(plot_names):
        # Extract raw columns
        col_train = X_train_for_metrics[:, j].astype(np.float32)
        col_test = X_test_for_metrics[:, j].astype(np.float32)
        
        # Calibrate using the TRAINING set of the best fold
        calibrated_scores = _calibrate_and_predict(col_train, y_train_for_metrics, col_test)
        y_test_for_metrics = y_test_for_metrics.astype(np.float32)
        # Calculate ROC on the PREDICTIONS
        fpr_m, tpr_m, _ = roc_curve(y_test_for_metrics, calibrated_scores)
        auc_m = auc(fpr_m, tpr_m)
        fold_tpr = _tpr_at_fpr(y_test_for_metrics, calibrated_scores, target_fpr=target_fpr)
        print(name, fold_tpr, auc_m)

        
        ax.plot(
            fpr_m, tpr_m,
            linestyle="--", linewidth=2,
            label=f"{name}{metrics_src_tag} (AUC={auc_m:.3f})"
        )

    # Chance line & Formatting
    ax.plot([1e-3, 1], [1e-3, 1], linestyle=":", linewidth=1, label="Chance", color='gray')
    ax.set_xscale("log")
    ax.set_xlim([1e-3, 1])
    ax.set_yscale("log")
    ax.set_ylim([1e-3, 1])
    ax.set_xlabel("False Positive Rate", fontsize=18)
    ax.set_ylabel("True Positive Rate", fontsize=18)
    
    final_title = title+" (Shadow)" if (X_shadow is not None) else title
    ax.set_title(final_title, fontsize=18)
    ax.legend(loc="lower right", fontsize=12)

    return unified_scores, test_idx, X_test_for_metrics
# -----------------------------
# Low-FPR analysis helpers
# -----------------------------
from scipy.stats import rankdata

def _threshold_at_target_fpr(scores: np.ndarray, y: np.ndarray, fpr_target: float) -> float:
    """Return score threshold that achieves the highest TPR with FPR <= fpr_target."""
    fpr, tpr, thr = roc_curve(y, scores)
    ok = fpr <= fpr_target
    if not np.any(ok):
        idx = np.argmin(np.abs(fpr - fpr_target))
        return thr[idx]
    idx = np.argmax(tpr[ok])
    return thr[np.where(ok)[0][idx]]

def _partial_auc(fpr: np.ndarray, tpr: np.ndarray, fpr_max: float) -> float:
    """Trapezoidal partial AUC over [0, fpr_max]."""
    mask = fpr <= fpr_max
    if not np.any(mask):
        return 0.0
    fpr_c = fpr[mask]
    tpr_c = tpr[mask]
    if fpr_c[-1] < fpr_max and len(fpr) > len(fpr_c):
        j = np.searchsorted(fpr, fpr_max)
        f0, f1 = fpr[j-1], fpr[j]
        t0, t1 = tpr[j-1], tpr[j]
        t_star = t0 + (t1 - t0) * (fpr_max - f0) / (f1 - f0)
        fpr_c = np.r_[fpr_c, fpr_max]
        tpr_c = np.r_[tpr_c, t_star]
    return np.trapz(tpr_c, fpr_c) / fpr_max if fpr_max > 0 else 0.0

def _ks_tail(pos_vals: np.ndarray, neg_vals: np.ndarray, high_tail: bool = True) -> float:
    pos = np.asarray(pos_vals, float)
    neg = np.asarray(neg_vals, float)
    ranks = rankdata(neg, method='average') / max(1, len(neg))
    qs = np.linspace(0.9, 1.0, 25) if high_tail else np.linspace(0.0, 0.1, 25)
    def cdf(x, arr): return (arr <= x).mean() if high_tail else (arr >= x).mean()
    ks = 0.0
    for q in qs:
        x = np.quantile(neg, q) if high_tail else np.quantile(neg, 1 - q)
        ks = max(ks, abs(cdf(x, pos) - cdf(x, neg)))
    return float(ks)

def _percentile_in_negatives(values: np.ndarray, negatives: np.ndarray) -> np.ndarray:
    neg = np.asarray(negatives, float)
    return 100.0 * (np.searchsorted(np.sort(neg), values, side='right') / max(1, len(neg)))

def _orient_scores_to_positive(s: np.ndarray, y: np.ndarray, flip_tol: float = 1e-6):
    fpr, tpr, _ = roc_curve(y, s)
    auc_raw = auc(fpr, tpr)
    flipped = False
    if np.isfinite(auc_raw) and (auc_raw < 0.5 - flip_tol):
        s = -s
        flipped = True
        auc_oriented = 1.0 - auc_raw
        fpr, tpr, _ = roc_curve(y, s)
    else:
        auc_oriented = auc_raw
    return s, flipped, float(auc_raw if np.isfinite(auc_raw) else 0.5), float(auc_oriented if np.isfinite(auc_oriented) else 0.5), fpr, tpr


def run_lowfpr_analysis(
    out_dir: Path,
    run_name: str,
    y: np.ndarray,
    unified_scores: np.ndarray,
    X_for_metrics: np.ndarray,
    feature_names: List[str],
    fpr_target: float = 1e-3,
    dump_tn: bool = False,
    best_fold_indices: Optional[np.ndarray] = None
):
    out_dir.mkdir(parents=True, exist_ok=True)

    # if best_fold_indices is not None:
    #     print(f"[LowFPR Analysis] Using data from the best fold ({len(best_fold_indices)} samples).")
    #     y_subset = y[best_fold_indices]
    #     unified_scores_subset = unified_scores  # unified_scores is already subset to best fold
    #     X_for_metrics_subset = X_for_metrics[best_fold_indices]
    # else:
    print("[LowFPR Analysis] WARNING: best_fold_indices not provided. Using full dataset.")
    y_subset = y
    unified_scores_subset = unified_scores
    X_for_metrics_subset = X_for_metrics

    thr = _threshold_at_target_fpr(unified_scores_subset, y_subset, fpr_target)
    y_hat = (unified_scores_subset >= thr).astype(int)

    TP_mask = (y_subset == 1) & (y_hat == 1)
    TN_mask = (y_subset == 0) & (y_hat == 0)

    rows = []
    flip_records = []
    X_oriented = X_for_metrics_subset.astype(float, copy=True)

    for j, name in enumerate(feature_names):
        s_raw = X_for_metrics_subset[:, j].astype(float)

        m = np.isfinite(s_raw)
        if not np.all(m):
            fill = np.nanmedian(s_raw) if np.any(m) else 0.0
            s_raw = np.where(m, s_raw, fill)

        s_or, flipped, auc_raw, auc_oriented, fpr_m, tpr_m = _orient_scores_to_positive(s_raw, y_subset)

        X_oriented[:, j] = s_or

        ok = (fpr_m <= fpr_target)
        if np.any(ok):
            tpr_low = float(np.max(tpr_m[ok]))
        else:
            tpr_low = float(tpr_m[np.argmin(np.abs(fpr_m - fpr_target))])

        pAUC = _partial_auc(fpr_m, tpr_m, fpr_target)
        pos_vals = s_or[y_subset == 1]
        neg_vals = s_or[y_subset == 0]
        ks_t = _ks_tail(pos_vals, neg_vals, high_tail=True)

        rows.append((name, tpr_low, pAUC, ks_t, int(flipped), auc_raw, auc_oriented))
        flip_records.append((name, int(flipped), auc_raw, auc_oriented))

    rows.sort(key=lambda r: (r[1], r[2], r[3]), reverse=True)

    import csv
    feat_csv = out_dir / f"lowfpr_feature_report_{run_name}.csv"
    with feat_csv.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "feature",
            f"TPR@FPR<={fpr_target}",
            f"pAUC[0,{fpr_target}]",
            "KS_tail",
            "flipped",
            "auc_raw",
            "auc_oriented"
        ])
        for r in rows:
            w.writerow(r)

    orient_csv = out_dir / f"lowfpr_orientation_{run_name}.csv"
    with orient_csv.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["feature", "flipped", "auc_raw", "auc_oriented"])
        for r in flip_records:
            w.writerow(r)

    neg_mat = X_oriented[y_subset == 0]
    tp_mat = X_oriented[TP_mask]
    tp_scores = unified_scores_subset[TP_mask]

    tp_csv = out_dir / f"lowfpr_tp_samples_{run_name}_best_fold_subset.csv"
    with tp_csv.open("w", newline="") as f:
        w = csv.writer(f)
        header = ["subset_index", "unified_score"] + \
                 [f"{fn}_value" for fn in feature_names] + \
                 [f"{fn}_pct_in_negs" for fn in feature_names]
        w.writerow(header)
        for i_loc in range(len(tp_scores)):
            vals = tp_mat[i_loc, :].astype(float)
            pcts = [_percentile_in_negatives(np.array([vals[j]]), neg_mat[:, j])[0] for j in range(len(feature_names))]
            w.writerow([i_loc, float(tp_scores[i_loc])] + list(map(float, vals)) + list(map(float, pcts)))

    if dump_tn:
        tn_mat = X_oriented[TN_mask]
        tn_scores = unified_scores_subset[TN_mask]
        tn_csv = out_dir / f"lowfpr_tn_samples_{run_name}_best_fold_subset.csv"
        with tn_csv.open("w", newline="") as f:
            w = csv.writer(f)
            header = ["subset_index", "unified_score"] + \
                     [f"{fn}_value" for fn in feature_names] + \
                     [f"{fn}_pct_in_negs" for fn in feature_names]
            w.writerow(header)
            for i_loc in range(len(tn_scores)):
                vals = tn_mat[i_loc, :].astype(float)
                pcts = [_percentile_in_negatives(np.array([vals[j]]), neg_mat[:, j])[0] for j in range(len(feature_names))]
                w.writerow([i_loc, float(tn_scores[i_loc])] + list(map(float, vals)) + list(map(float, pcts)))


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--npz",
        type=Path,
        nargs="+",
        required=True,
        help="One or more *directories* whose 'metrics_prompt.npz' should be used as TARGET.",
    )
    ap.add_argument(
        "--labels",
        type=str,
        nargs="+",
        default=None,
        help="Optional subplot titles, must match the number of --npz files.",
    )
    ap.add_argument(
        "--drop",
        type=str,
        nargs="+",
        default=None,
        help="Metric names to DROP (match ORIGINAL names; repeatable).",
    )
    ap.add_argument(
        "--rename",
        type=str,
        nargs="+",
        default=None,
        help="Rename pairs 'old=new' (affects display names). Repeatable.",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=Path("tpr_fpr_multi.png"),
        help="Output image path (.png or .pdf)",
    )
    ap.add_argument(
        "--title",
        type=str,
        default="ROC (TPR vs FPR): Best Fold Unified vs Individual (Multiple Runs)",
    )
    ap.add_argument(
        "--cols",
        type=int,
        default=2,
        help="Number of subplot columns (rows auto-computed).",
    )
    ap.add_argument(
        "--shadow-npz",
        type=Path,
        nargs="+",
        default=None,
        help="Optional SHADOW model directories (each containing 'metrics_prompt.npz'). "
             "If one path is given, it is broadcast to all target runs.",
    )
    ap.add_argument(
        "--align-key",
        type=str,
        default=None,
        help=f"Optional attribute name to align prompts by (e.g., one of {CANDIDATE_KEY_ATTRS}). "
             "If omitted, auto-detect; fallback to index-based with warning if no IDs available.",
    )
    ap.add_argument("--fpr-target", type=float, default=1e-3,
                    help="Operating FPR for low-FPR analysis.")
    ap.add_argument("--analysis-out-dir", type=Path, default=Path("lowfpr_analysis"),
                    help="Directory to write CSV analysis outputs.")
    ap.add_argument("--dump-tn", action="store_true", help="Also dump TN samples table.")

    args = ap.parse_args()

    npz_paths = [Path.joinpath(x, 'metrics_prompt.npz') for x in args.npz]
    if args.labels is not None and len(args.labels) != len(npz_paths):
        raise ValueError("--labels count must match the number of --npz paths")

    shadow_paths = None
    if args.shadow_npz:
        if len(args.shadow_npz) == 1 and len(npz_paths) > 1:
            shadow_paths = [args.shadow_npz[0]] * len(npz_paths)
        else:
            if len(args.shadow_npz) != len(npz_paths):
                raise ValueError("--shadow-npz count must be 1 or equal to the number of --npz paths")
            shadow_paths = list(args.shadow_npz)
        shadow_paths = [Path.joinpath(x, 'metrics_prompt.npz') for x in shadow_paths]

    rename_map = parse_renames(args.rename)
    drop_set = set(args.drop or [])
    specs = _metric_specs(rename_map=rename_map, drop_set=drop_set)
    if not specs:
        raise ValueError("After applying --drop, no metrics remain to plot.")

    prepared = []
    for i, pth in enumerate(npz_paths):
        if not pth.exists():
            raise FileNotFoundError(f"TARGET NPZ not found: {pth}")
        prompts_t = load_prompt_data_from_npz(pth)

        key_attr = args.align_key or _detect_key_attr(prompts_t)
        if key_attr:
            print(f"[INFO] Using alignment key '{key_attr}' for TARGET run {i}.")
        else:
            print(f"[WARN] Could not detect an alignment key for TARGET run {i}; "
                  "will fallback to index-based if needed.")

        X_t, y_t, plot_names, keys_t = _build_matrix_and_labels(prompts_t, specs, return_keys=True, key_attr=key_attr)
        title = args.labels[i] if (args.labels and i < len(args.labels)) else pth.parent.name

        X_s, y_s = None, None
        if shadow_paths is not None:
            sp = shadow_paths[i]
            if not sp.exists():
                raise FileNotFoundError(f"SHADOW NPZ not found: {sp}")
            prompts_s = load_prompt_data_from_npz(sp)

            key_attr_s = args.align_key or _detect_key_attr(prompts_s)
            if key_attr and key_attr_s and key_attr != key_attr_s:
                print(f"[WARN] Mismatched detected keys (target='{key_attr}', shadow='{key_attr_s}'); "
                      f"will still try '{args.align_key or key_attr}' on both.")
                key_attr_s = key_attr

            X_s_raw, y_s_raw, _, keys_s = _build_matrix_and_labels(prompts_s, specs, return_keys=True, key_attr=key_attr_s)

            X_t_aln, y_t_aln, X_s_aln, y_s_aln, _ = _align_by_keys(
                X_t, y_t, keys_t, X_s_raw, y_s_raw, keys_s
            )

            X_t, y_t, X_s, y_s = X_t_aln, y_t_aln, X_s_aln, y_s_aln

            if len(np.unique(y_t)) < 2:
                raise ValueError(f"After alignment, TARGET run {i} has fewer than 2 classes.")

        prepared.append((X_t, y_t, plot_names, title, X_s, y_s))

    n = len(prepared)
    ncols = max(1, args.cols)
    nrows = math.ceil(n / ncols)

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(6.5*ncols, 6.5*nrows))

    if nrows * ncols == 1:
        axes = np.array([[axes]])
    elif nrows == 1:
        axes = np.array([axes])
    elif ncols == 1:
        axes = np.array([[ax] for ax in axes])

    axes_colmajor = axes.T.flatten()

    for (X_t, y_t, plot_names, ttl, X_s, y_s), ax in zip(prepared, axes_colmajor):
        unified_scores, best_fold_indices, X_for_metrics_used = _plot_single_axes(
            ax, X_t, y_t, plot_names, ttl, random_state=CONF.random_state,
            X_shadow=X_s, y_shadow=y_s
        )

        # run_lowfpr_analysis(
        #     out_dir=args.analysis_out_dir,
        #     run_name=ttl.replace(" ", "_"),
        #     y=y_t,
        #     unified_scores=unified_scores,
        #     X_for_metrics=X_for_metrics_used,
        #     feature_names=plot_names,
        #     fpr_target=args.fpr_target,
        #     dump_tn=args.dump_tn,
        #     best_fold_indices=best_fold_indices
        # )

    for k in range(len(prepared), len(axes_colmajor)):
        axes_colmajor[k].axis("off")

    fig.tight_layout(rect=[0, 0.02, 1, 0.97])

    args.out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.out, dpi=200, bbox_inches="tight")
    print(f"Saved multi-run ROC figure (Best Fold per Run) to {args.out}")


if __name__ == "__main__":
    main()