# evaluate_individual_metrics.py
import argparse
import json
from pathlib import Path
import numpy as np
# Changed: Import StratifiedKFold instead of train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# Ensure the script can find the 'src' module
import sys
sys.path.append(str(Path(__file__).parent))

from src.utils import load_prompt_data_from_npz, PromptData
from src.evaluation import get_metrics
from src.config import CONF

def evaluate_single_feature(
    X_feature: np.ndarray, 
    y: np.ndarray,
    n_splits: int = 5
) -> dict:
    """
    Trains and evaluates a simple logistic regression model on a single feature
    using Stratified K-Fold Cross-Validation.

    Args:
        X_feature: A numpy array of shape (n_samples, 1) representing the single feature.
        y: The corresponding labels.
        n_splits: Number of folds (default 5).

    Returns:
        A dictionary containing averaged performance metrics (AUC, TPR@FPR, etc.).
    """
    
    # Initialize Stratified K-Fold
    skf = StratifiedKFold(
        n_splits=n_splits, 
        shuffle=True, 
        random_state=CONF.random_state
    )

    fold_metrics_list = []

    # Loop through the folds
    for fold_idx, (train_index, test_index) in enumerate(skf.split(X_feature, y)):
        X_tr, X_te = X_feature[train_index], X_feature[test_index]
        y_tr, y_te = y[train_index], y[test_index]

        # Handle cases with insufficient data for training in this fold
        if len(np.unique(y_tr)) < 2 or len(X_tr) == 0:
            continue # Skip this fold if valid training isn't possible
        
        # Clean potential NaN/inf values
        X_tr = np.nan_to_num(X_tr, nan=0.0, posinf=0.0, neginf=0.0)
        X_te = np.nan_to_num(X_te, nan=0.0, posinf=0.0, neginf=0.0)

        # Create and train the model
        model = make_pipeline(StandardScaler(), LogisticRegression(random_state=CONF.random_state))
        model.fit(X_tr, y_tr)

        # Predict probabilities and labels on the test set
        y_prob = model.predict_proba(X_te)[:, 1]
        y_pred = (y_prob >= 0.5).astype(int)
        
        # Calculate metrics for this fold
        metrics = get_metrics(y_te, y_pred, y_prob)
        fold_metrics_list.append(metrics)

    # If no folds succeeded, return error
    if not fold_metrics_list:
        return {"error": "Not enough data or classes to train in any fold."}

    # --- Aggregate (Average) Metrics across folds ---
    averaged_results = {}
    
    # Get all keys from the first successful fold
    metric_keys = fold_metrics_list[0].keys()

    for key in metric_keys:
        # Collect values for this key from all folds
        values = [m[key] for m in fold_metrics_list if isinstance(m[key], (int, float))]
        
        if values:
            # Calculate mean
            averaged_results[key] = float(np.mean(values))
        else:
            # If it's a string (like an error message), just take the first one
            averaged_results[key] = fold_metrics_list[0][key]

    return averaged_results

def main():
    """
    Main function to load data and run the evaluation for each feature.
    """
    parser = argparse.ArgumentParser(
        description="Evaluate the stand-alone performance of each individual metric (5-Fold CV)."
    )
    parser.add_argument(
        "--load_from_npz", 
        required=True, 
        type=Path, 
        help="Path to the metrics_prompt.npz file to load for evaluation."
    )
    parser.add_argument(
        "--out_dir", 
        required=True, 
        type=Path, 
        help="Directory to save the performance report."
    )
    args = parser.parse_args()

    # Create output directory if it doesn't exist
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Check if the input file exists
    if not args.load_from_npz.exists():
        raise FileNotFoundError(f"Cache file not found: {args.load_from_npz}")

    print(f"Loading data from {args.load_from_npz}...")
    prompt_data = load_prompt_data_from_npz(args.load_from_npz)
    
    # Prepare the full feature matrix and labels
    X_full = np.array([p.to_feature_array() for p in prompt_data])
    y = np.array([p.label for p in prompt_data])
    feature_names = PromptData.get_all_feature_names()
    
    all_results = {}

    print("\n--- Evaluating Stand-alone Performance (5-Fold CV) ---")
    # Iterate through each feature, evaluate it, and store the results
    for i, feature_name in enumerate(feature_names):
        print(f"Evaluating: {feature_name}...", end=" ", flush=True)
        
        # Isolate the current feature column
        X_single_feature = X_full[:, i].reshape(-1, 1)
        
        # Run the evaluation (Default is 5-fold)
        metrics = evaluate_single_feature(X_single_feature, y, n_splits=5)
        all_results[feature_name] = metrics
        
        auc_val = metrics.get('auc', 'N/A')
        if isinstance(auc_val, float):
            print(f"Mean AUC: {auc_val:.4f}")
        else:
            print(f"Result: {auc_val}")

    # --- Report Generation ---
    report_path = args.out_dir / "report_individual_metric_performance_5fold.json"
    print(f"\nSaving detailed report to {report_path}...")
    with open(report_path, 'w') as f:
        json.dump(all_results, f, indent=4)

    # --- Summary Report ---
    summary_path = args.out_dir / "summary_individual_metric_performance_5fold.txt"
    print(f"Saving summary report to {summary_path}...")
    # Sort metrics by AUC score in descending order for the summary
    sorted_metrics = sorted(all_results.items(), key=lambda item: item[1].get('auc', 0), reverse=True)
    
    with open(summary_path, 'w') as f:
        f.write("--- Stand-alone Metric Performance Summary (5-Fold CV Average) ---\n\n")
        f.write(f"{'Feature Name':<30} | {'Mean AUC':<10} | {'Mean F1':<10} | {'TPR@0.1%':<10}\n")
        f.write("-" * 75 + "\n")
        
        for feature_name, metrics in sorted_metrics:
            auc = metrics.get('auc', float('nan'))
            f1 = metrics.get('f1_score', float('nan'))
            tpr_1 = metrics.get('tpr_at_0.1%_fpr', float('nan'))
            
            # Format handling for errors
            auc_str = f"{auc:.4f}" if isinstance(auc, (int, float)) else "Err"
            f1_str = f"{f1:.4f}" if isinstance(f1, (int, float)) else "Err"
            tpr_str = f"{tpr_1:.4f}" if isinstance(tpr_1, (int, float)) else "Err"

            f.write(f"{feature_name:<30} | {auc_str:<10} | {f1_str:<10} | {tpr_str:<10}\n")

    print("\nEvaluation complete.")

if __name__ == "__main__":
    main()
'''
python indivdual.py \
    --load_from_npz math_dapo_3b_temp/t=0.5/metrics_prompt.npz\
    --out_dir math_dapo_3b_temp/t=0.5
    '''
'''
python indivdual.py \
    --load_from_npz out_runs/exp_traintest/global_step_300/metrics_prompt.npz\
    --out_dir out_runs/exp_traintest/global_step_300
'''