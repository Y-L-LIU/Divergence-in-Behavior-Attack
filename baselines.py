#!/usr/bin/env python3
"""
Membership Inference (QA) with 5-Fold Cross-Validation
"""
import argparse, csv, json, math, os, random, re, zlib
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import io
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import roc_auc_score
# NEW IMPORT
from sklearn.model_selection import StratifiedKFold
import numpy as np
from transformers import (
    AutoTokenizer,
    AutoProcessor,
    AutoModelForCausalLM,
    AutoModelForMaskedLM,
    pipeline as hf_pipeline,
)

# ... [Keep all Config, Metrics helpers, Input builder, NeighborGen classes exactly as they were] ...
# ... [Make sure to keep evaluate_metric_on_test and other helper functions] ...

# ----------------------------
# (Paste the Config, Helper classes, and Functions here from your original script)
# For brevity, I am assuming the helper functions (compute_mia_features, evaluate_metric_on_test, etc.) 
# are defined above as they were in your provided code.
# ----------------------------

# [Restoring necessary imports and helpers for the context of the solution]
@dataclass
class MIAConf:
    temperature: float = 0.7
    neighbors: int = 0
    neigh_per_position: int = 5
    mlm_topk: int = 20
    seed: int = 42

def _compute_token_metrics(logits: torch.Tensor, token_ids: torch.Tensor, temperature: float = 0.7) -> Dict[str, Any]:
    temp_logits = logits / temperature
    logp = F.log_softmax(temp_logits, dim=-1)
    probs = logp.exp()
    token_lp_actual = torch.gather(logp, -1, token_ids.unsqueeze(-1)).squeeze(-1)
    neg_ll = -token_lp_actual.mean().item()
    entropy = -(probs * logp).sum(-1).mean().item()
    mu = (probs * logp).sum(-1)
    var = (probs * logp.pow(2)).sum(-1) - mu.pow(2)
    z_scores = (token_lp_actual - mu) / var.clamp_min(1e-12).sqrt()
    k = max(1, int(z_scores.numel() * 0.2))
    minkowski_score = z_scores.topk(k, largest=False).values.mean().item()
    return {"neg_ll": neg_ll, "entropy": entropy, "minkowski": minkowski_score}

def _build_inputs(prompt, response, image, tok, device):
    # (Same as your original code)
    if image is not None:
        if not prompt.endswith("\n"): prompt = prompt + "\n"
        msg1 = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": prompt}]}]
        cond = tok.apply_chat_template(msg1, add_generation_prompt=True, tokenize=False)
        full_chat = cond + response
        inputs_prompt = tok(text=[cond], images=[image], return_tensors="pt")
        inputs_full   = tok(text=[full_chat], images=[image], return_tensors="pt")
        inputs_prompt = {k: v.to(device) for k, v in inputs_prompt.items()}
        inputs_full   = {k: v.to(device)   for k, v in inputs_full.items()}
        full_ids      = inputs_full["input_ids"]
        prompt_len    = int(inputs_prompt["input_ids"].shape[1])
        return inputs_full, full_ids, prompt_len
    else:
        prompt_ids = tok(prompt, return_tensors="pt").input_ids.to(device)
        resp_ids   = tok(response, return_tensors="pt").input_ids.to(device)
        full_ids   = torch.cat([prompt_ids, resp_ids], dim=1).to(device)
        attn_mask  = torch.ones_like(full_ids)
        inputs_full = {"input_ids": full_ids, "attention_mask": attn_mask}
        return inputs_full, full_ids, int(prompt_ids.shape[1])

# ... [Assume NeighborGen and other helpers are here] ...
MASKABLE_RE = re.compile(r"[A-Za-z]+(?:[-'][A-Za-z]+)?")
# (Assuming the rest of the helpers exist as in your prompt)
def evaluate_metric_on_test(rows: List[Dict[str, Any]], score_key: str, target_fpr: float = 0.001) -> Dict[str, float]:
    # (Same logic as your code, relies on 'split' key in rows)
    test_rows = [r for r in rows if r.get("split") == "test" and math.isfinite(r[score_key])]
    if not test_rows: return {}
    raw_scores = [r[score_key] for r in test_rows]
    labels = [r["member"] for r in test_rows]
    try:
        auc_inverted = roc_auc_score(labels, [-s for s in raw_scores])
    except ValueError: return {}
    
    if auc_inverted < 0.5:
        scores = raw_scores; auc = 1.0 - auc_inverted; direction = "higher"
    else:
        scores = [-s for s in raw_scores]; auc = auc_inverted; direction = "lower"

    test_mem_scores = [s for s, l in zip(scores, labels) if l == 1]
    test_non_scores = [s for s, l in zip(scores, labels) if l == 0]
    n_mem, n_non = len(test_mem_scores), len(test_non_scores)
    if n_non == 0 or n_mem == 0: return {}

    sorted_non = sorted(test_non_scores, reverse=True)
    k = max(0, min(len(sorted_non) - 1, int(math.floor(target_fpr * len(sorted_non)))))
    tau_corrected = sorted_non[k]
    achieved_fpr = sum(1 for s in test_non_scores if s >= tau_corrected) / n_non
    tpr = sum(1 for s in test_mem_scores if s >= tau_corrected) / n_mem
    tau_original = -tau_corrected if direction == "lower" else tau_corrected

    return {"AUC": float(auc), "TPR@FPR": float(tpr), "tau": float(tau_original), 
            "achieved_FPR": float(achieved_fpr), "n_test_non": n_non, "n_test_mem": n_mem}

def norm_df(df: pd.DataFrame, label: int) -> pd.DataFrame:
    # Minimal version for context
    if "extra_info" in df.columns:
        # (Assuming your JSON parsing logic is here)
        pass
    return pd.DataFrame({"prompt": df["prompt"], "answer": df["answer"], "member": int(label), "images": None})

def _parse_extra_info(x): return {} # Placeholder
def load_tokenizer_or_processor(name): return AutoTokenizer.from_pretrained(name, trust_remote_code=True)
def load_causal_model(name, dev, dtype): return AutoModelForCausalLM.from_pretrained(name, trust_remote_code=True).to(dev)
def load_image(img): return None

# ----------------------------
# MAIN MODIFICATION STARTS HERE
# ----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--member", required=True, help="Path to member.parquet")
    ap.add_argument("--nonmember", required=True, help="Path to nonmember.parquet")
    # ... [Keep existing args] ...
    ap.add_argument("--prompt-col", default="prompt")
    ap.add_argument("--answer-col", default="answer")
    ap.add_argument("--image-col", default="image")
    ap.add_argument("--target-fpr", type=float, default=0.001)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--model-ft", required=True)
    ap.add_argument("--model-ref", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default=None)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--mlm", default=None)
    ap.add_argument("--neighbors", type=int, default=0)
    ap.add_argument("--mlm-topk", type=int, default=20)
    ap.add_argument("--neigh-per-pos", type=int, default=5)
    ap.add_argument("--out", required=True, help="CSV with per-sample features")
    ap.add_argument("--metrics-out", default=None)
    
    # NEW ARGUMENT
    ap.add_argument("--k-folds", type=int, default=5, help="Number of folds for Cross Validation")

    args = ap.parse_args()
    torch.set_grad_enabled(False)
    random.seed(args.seed)
    np.random.seed(args.seed)

    if args.metrics_out is None:
        base = os.path.splitext(args.out)[0]
        args.metrics_out = f"{base}_metrics_5fold.csv"

    # 1. LOAD OR COMPUTE FEATURES
    rows = []
    
    if os.path.exists(args.out):
        print(f"Loading existing features from {args.out}")
        df = pd.read_csv(args.out)
        rows = df.to_dict('records')
        print(f"Loaded {len(rows)} samples from cache.")
    else:
        # (Data loading and model inference logic remains the same)
        # ... [Load parquets, normalize, concat] ...
        # For brevity in this answer, assume df is prepared here
        # Note: We do NOT need to do a train/test split here anymore for the sake of calculation,
        # but the code below assumes 'split' key might be written to CSV.
        
        # Placeholder for actual data loading logic to make this snippet runnable logic-wise:
        # df_mem = pd.read_parquet(args.member); df_non = pd.read_parquet(args.nonmember) ...
        # ... computation loop ...
        pass
        # IMPORTANT: When saving to CSV in the "else" block, you can just set "split" to "N/A" 
        # or a dummy value, because we will overwrite it during 5-fold evaluation.

    # Ensure rows have integer indices for splitting
    for idx, r in enumerate(rows):
        r['_temp_idx'] = idx

    # 2. PREPARE 5-FOLD CV
    print(f"\n--- Starting {args.k_folds}-Fold Cross-Validation ---")
    
    # Extract labels for stratification
    y = np.array([r["member"] for r in rows])
    indices = np.array(range(len(rows)))

    skf = StratifiedKFold(n_splits=args.k_folds, shuffle=True, random_state=args.seed)

    # Dictionary to aggregate results per metric
    # Structure: { "metric_name": { "AUC": [val1, val2...], "TPR@FPR": [...] } }
    aggregated_metrics = {}

    metrics_for_inference = [
        "LOSS", "lira", "Zlib", "Zlib (Cali)", 
        "Min-K%", "Min-K% (Cali)", "Entropy", "Entropy (Cali)"
    ]
    if any("nc_loss_diff" in r for r in rows):
        metrics_for_inference.append("nc_loss_diff")

    # Initialize storage
    for m in metrics_for_inference:
        aggregated_metrics[m] = {"AUC": [], "TPR@FPR": []}

    fold_idx = 0
    for train_index, test_index in skf.split(indices, y):
        fold_idx += 1
        print(f"Processing Fold {fold_idx}/{args.k_folds}...")

        # Update the 'split' key in rows for this specific fold
        # optimize: convert test_index to set for O(1) lookup
        test_set_indices = set(test_index)
        
        for i in range(len(rows)):
            if i in test_set_indices:
                rows[i]["split"] = "test"
            else:
                rows[i]["split"] = "train"

        # Evaluate current fold
        for key in metrics_for_inference:
            res = evaluate_metric_on_test(rows, score_key=key, target_fpr=args.target_fpr)
            
            if "AUC" in res: # valid result
                aggregated_metrics[key]["AUC"].append(res["AUC"])
                aggregated_metrics[key]["TPR@FPR"].append(res["TPR@FPR"])

    # 3. COMPUTE STATISTICS AND SAVE
    final_report = []
    
    print("\n=== Membership Inference Report (5-Fold CV Average) ===")
    print(f"{'Metric':>20} | {'AUC (Mean ± Std)':>20} | {f'TPR@{args.target_fpr} FPR (Mean ± Std)':>30}")
    print("-" * 80)

    for key in metrics_for_inference:
        aucs = aggregated_metrics[key]["AUC"]
        tprs = aggregated_metrics[key]["TPR@FPR"]
        
        if not aucs:
            continue

        mean_auc = np.mean(aucs)
        std_auc = np.std(aucs)
        mean_tpr = np.mean(tprs)
        std_tpr = np.std(tprs)

        print(f"{key:>20} | {mean_auc:.4f} ± {std_auc:.4f} | {mean_tpr:.4f} ± {std_tpr:.4f}")

        final_report.append({
            "metric": key,
            "mean_auc": mean_auc,
            "std_auc": std_auc,
            "mean_tpr": mean_tpr,
            "std_tpr": std_tpr,
            "target_fpr": args.target_fpr,
            "k_folds": args.k_folds
        })

    # Save summary to CSV
    metrics_df = pd.DataFrame(final_report)
    metrics_df.to_csv(args.metrics_out, index=False)
    print(f"\nSaved averaged 5-fold metric results to: {args.metrics_out}")

if __name__ == "__main__":
    main()