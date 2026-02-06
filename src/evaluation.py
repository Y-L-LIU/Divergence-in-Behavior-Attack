# mia_project/evaluation.py

import json
from typing import List, Tuple, Dict, Any
import numpy as np
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from collections import defaultdict
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

# Import from our other project modules
from .config import CONF
from .utils import tpr_at_fpr, write_report, clean_infs,PromptData
import csv

def write_summary_csv(
    out_dir: Path,
    unified_summary: Dict[str, str],
    individual_specialist_summary: Dict[str, Dict[str, str]],
    combined_specialist_summary: Dict[str, str],
    sample_counts: Dict[str, float]
):
    """
    Writes evaluation results to a CSV file in the specified format, including average # samples.
    """
    csv_path = out_dir / "evaluation_summary.csv"
    
    def safe_extract(summary_dict, key_pattern):
        for k, v in summary_dict.items():
            if key_pattern in k:
                try:
                    mean_str = v.split(' ± ')[0]
                    return float(mean_str)
                except:
                    return float('nan')
        return float('nan')
    
    rows = []
    
    # Map internal names to display names
    name_map = {
        "all-0": "ALL-0",
        "all-1": "ALL-1",
        "residual": "Residual"
    }
    
    for internal_name in ["all-0", "all-1", "residual"]:
        display_name = name_map[internal_name]
        summary = individual_specialist_summary.get(internal_name, {})
        auc = safe_extract(summary, "auc")
        tpr1 = safe_extract(summary, "tpr_at_1.0%_fpr")
        tpr2 = safe_extract(summary, "tpr_at_0.1%_fpr")

        avg_samples = sample_counts.get(internal_name, 0)
        rows.append([display_name, f"{avg_samples:.0f}", f"{auc:.3f}", f"{tpr1:.3f}", f"{tpr2:.3f}"])
    
    # Unified model
    auc_unified = safe_extract(unified_summary, "auc")
    tpr_unified1 = safe_extract(unified_summary, "tpr_at_1.0%_fpr")
    tpr_unified2 = safe_extract(unified_summary, "tpr_at_0.1%_fpr")

    avg_samples_unified = sample_counts.get("unified", 0)
    rows.append(["Unified", f"{avg_samples_unified:.0f}", f"{auc_unified:.3f}", f"{tpr_unified1:.3f}",f"{tpr_unified2:.3f}"])
    
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["Split", "GRPO", "", ""])
        writer.writerow(["", "# Samples", "AUC", "TPR @ 1.0% FPR","TPR @ 0.1% FPR" ])
        for row in rows:
            writer.writerow(row)
    
    print(f"CSV summary with sample counts written to: {csv_path}")

def get_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
    """Calculates a dictionary of performance metrics."""
    y_prob = np.nan_to_num(y_prob) # Clean probabilities
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_score": f1_score(y_true, y_pred, zero_division=0)
    }
    if len(np.unique(y_true)) >= 2:
        metrics["auc"] = roc_auc_score(y_true, y_prob)
        for fpr_target in CONF.target_fprs:
            metrics[f"tpr_at_{fpr_target*100}%_fpr"] = tpr_at_fpr(y_true, y_prob, fpr_target)
    else:
        metrics["auc"] = float('nan')
        for fpr_target in CONF.target_fprs:
            metrics[f"tpr_at_{fpr_target*100}%_fpr"] = float('nan')
    return metrics

def train_and_evaluate_model(X_tr: np.ndarray, y_tr: np.ndarray, X_te: np.ndarray, y_te: np.ndarray) -> Tuple[Any, Dict[str, float]]:
    """
    Trains a stacking classifier on the training set and evaluates it on the test set.
    This function does NOT perform any data splitting.
    """
    if len(np.unique(y_tr)) < 2 or len(X_tr) == 0:
        return None, {"error": "Not enough data or classes to train."}

    X_tr, X_te = clean_infs(X_tr), clean_infs(X_te)
    
    # Feature selection based ONLY on training data
    X_tr_sel, X_te_sel = X_tr, X_te
    
    estimators = [
        ('rf', RandomForestClassifier(n_estimators=100, random_state=CONF.random_state)),
        ('lr', make_pipeline(StandardScaler(), LogisticRegression(random_state=CONF.random_state)))
    ]
    model = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression(), cv=5)
    
    model.fit(X_tr_sel, y_tr)
    
    # Evaluate on the provided test set
    y_prob = model.predict_proba(X_te_sel)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)
    
    metrics = get_metrics(y_te, y_pred, y_prob)
    
    return model, metrics
def _summarize_fold_metrics(metrics_list: List[Dict[str, float]]) -> Dict[str, str]:
    """
    Aggregates a list of metric dictionaries from k-folds into a summary dictionary.
    Calculates the mean and standard deviation for each metric.
    """
    if not metrics_list:
        return {"status": "No data to summarize."}
        
    summary = {}
    metric_keys = metrics_list[0].keys()
    
    for key in metric_keys:
        values = [m[key] for m in metrics_list if key in m]
        if values:
            mean = np.mean(values)
            std = np.std(values)
            summary[key] = f"{mean:.4f} ± {std:.4f}"
            
    return summary

def run_evaluation(prompts: List[PromptData], out_dir: Path):
    """
    Runs and reports on two evaluation regimes using 5-fold cross-validation.
    """
    print("\n--- Starting 5-Fold Cross-Validation Evaluation ---")
    
    X = np.array([p.to_feature_array() for p in prompts])
    y = np.array([p.label for p in prompts])
    
    # if np.isnan(X).any():
    #     print(f"WARNING: Found {np.isnan(X).sum()} NaN values in feature matrix. Imputing with 0.")
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=CONF.random_state)
    
    unified_fold_metrics = []
    specialist_combined_fold_metrics = []
    specialist_individual_fold_metrics = defaultdict(list)
    
    # NEW: Track test sample counts per fold for each split
    unified_test_counts = []
    specialist_test_counts = defaultdict(list)  # e.g., {"all-0": [10, 12, ...], ...}

    all_prompts_data = np.array(prompts)
    all_zero_flags = np.array([abs(p.avg_base_score) < CONF.score_tolerance and abs(p.avg_ft_score) < CONF.score_tolerance for p in all_prompts_data])
    all_one_flags = np.array([abs(p.avg_base_score - 1.0) < CONF.score_tolerance and  abs(p.avg_ft_score-1) < CONF.score_tolerance for p in all_prompts_data])
    
    masks = {
        "all-0": all_zero_flags,
        "all-1": all_one_flags,
        "residual": ~all_zero_flags & ~all_one_flags
    }

    for fold_idx, (train_indices, test_indices) in enumerate(skf.split(X, y)):
        print(f"\n===== Processing Fold {fold_idx + 1}/5 =====")
        X_tr, X_te = X[train_indices], X[test_indices]
        y_tr, y_te = y[train_indices], y[test_indices]

        # --- Regime 1: Unified Model ---
        print("\n[Regime 1: Unified Model]")
        unified_test_counts.append(len(y_te))  # Track unified test size
        _, unified_metrics = train_and_evaluate_model(X_tr, y_tr, X_te, y_te)
        unified_fold_metrics.append(unified_metrics)

        # # --- Regime 2: Specialist Models ---
        # print("\n[Regime 2: Specialist Models]")
        # y_prob_overall_fold = np.zeros_like(y_te, dtype=float)

        # for name, full_dataset_mask in masks.items():
        #     test_mask = full_dataset_mask[test_indices]
        #     test_count = np.sum(test_mask)
            
        #     # Track test sample count for this specialist in this fold
        #     specialist_test_counts[name].append(test_count)

        #     train_mask = full_dataset_mask[train_indices]
        #     print(f"  -> Fold {fold_idx+1}, Specialist '{name}': Train count={np.sum(train_mask)}, Test count={test_count}")

        #     if not np.any(train_mask):
        #         print(f"     Skipping '{name}': No training data in this fold.")
        #         continue

        #     # Evaluate only if there's test data
        #     if np.any(test_mask):
        #         specialist_model, metrics = train_and_evaluate_model(
        #             X_tr[train_mask], y_tr[train_mask],
        #             X_te[test_mask], y_te[test_mask]
        #         )
        #         specialist_individual_fold_metrics[name].append(metrics)
        #         if specialist_model:
        #             y_prob_overall_fold[test_mask] = specialist_model.predict_proba(X_te[test_mask])[:, 1]
        #     else:
        #         # No test data → no metrics, but model may still be trained (not used here)
        #         pass

        # Combined specialist metrics for this fold
        # y_pred_overall_fold = (y_prob_overall_fold >= 0.5).astype(int)
        # overall_specialist_metrics = get_metrics(y_te, y_pred_overall_fold, y_prob_overall_fold)
        # specialist_combined_fold_metrics.append(overall_specialist_metrics)

    # --- Compute average test sample counts ---
    print("\n--- Aggregating Results and Generating Final Reports ---")

    unified_summary = _summarize_fold_metrics(unified_fold_metrics)
    write_report(
        out_dir / "report_unified_model.txt",
        f"--- Unified Model Performance (5-Fold CV) ---\n{json.dumps(unified_summary, indent=2)}"
    )

    avg_sample_counts = {
        "unified": np.mean(unified_test_counts) if unified_test_counts else 0,
        "all-0": np.mean(specialist_test_counts["all-0"]) if specialist_test_counts["all-0"] else 0,
        "all-1": np.mean(specialist_test_counts["all-1"]) if specialist_test_counts["all-1"] else 0,
        "residual": np.mean(specialist_test_counts["residual"]) if specialist_test_counts["residual"] else 0,
    }

    # --- Final Report Generation ---
    print("\n--- Aggregating Results and Generating Final Reports ---")

    unified_summary = _summarize_fold_metrics(unified_fold_metrics)
    write_report(
        out_dir / "report_unified_model.txt",
        f"--- Unified Model Performance (5-Fold CV) ---\n{json.dumps(unified_summary, indent=2)}"
    )

    individual_specialist_summary = {
        name: _summarize_fold_metrics(metrics_list)
        for name, metrics_list in specialist_individual_fold_metrics.items()
    }
    write_report(
        out_dir / "report_specialist_models_individual.txt",
        f"--- Individual Specialist Model Performance (5-Fold CV) ---\n{json.dumps(individual_specialist_summary, indent=2)}"
    )

    combined_specialist_summary = _summarize_fold_metrics(specialist_combined_fold_metrics)
    write_report(
        out_dir / "report_specialist_models_combined.txt",
        f"--- Combined Specialist Model Performance (5-Fold CV) ---\n{json.dumps(combined_specialist_summary, indent=2)}"
    )

    # --- Generate CSV Summary with # Samples ---
    write_summary_csv(
        out_dir,
        unified_summary,
        individual_specialist_summary,
        combined_specialist_summary,
        avg_sample_counts
    )

    print("\n--- Evaluation Complete ---")