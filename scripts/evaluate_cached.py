"""Evaluate the numeric feature matrix as stored, including legacy five-column caches."""
from pathlib import Path
import argparse
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from sklearn.model_selection import StratifiedKFold
from src.evaluation import train_and_evaluate_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--limit_per_class", type=int, help="Small execution check; not a paper result")
    args = parser.parse_args()
    data = np.load(args.input, allow_pickle=False)
    X, y = data["feats"], data["labels"]
    if X.ndim != 2 or y.ndim != 1 or len(X) != len(y):
        raise ValueError("Expected aligned feats[N,D] and labels[N].")
    if args.limit_per_class:
        if args.limit_per_class < 7:
            parser.error("--limit_per_class must be at least 7 for nested five-fold CV.")
        chosen = np.concatenate([np.flatnonzero(y == label)[:args.limit_per_class] for label in (0, 1)])
        X, y = X[chosen], y[chosen]
    classes, counts = np.unique(y, return_counts=True)
    if set(classes) != {0, 1} or counts.min() < 7:
        raise ValueError("Need at least 7 prompts per membership class.")
    nonfinite = int((~np.isfinite(X)).sum())
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    folds = []
    for train, test in StratifiedKFold(5, shuffle=True, random_state=2025).split(X, y):
        _, metrics = train_and_evaluate_model(X[train], y[train], X[test], y[test])
        folds.append(metrics)
    report = {
        "input": str(args.input), "feature_shape": list(X.shape),
        "nonfinite_entries_replaced_with_zero": nonfinite,
        "limit_per_class": args.limit_per_class,
        "metrics": {name: {"mean": float(np.mean([f[name] for f in folds])),
                           "std": float(np.std([f[name] for f in folds]))} for name in folds[0]},
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "reevaluation.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
