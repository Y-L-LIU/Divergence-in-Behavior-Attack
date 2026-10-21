# utils.py
from pathlib import Path
import numpy as np
import json
from sklearn.metrics import roc_curve
from typing import Dict, List, Optional
from collections import defaultdict
import pandas as pd
from dataclasses import dataclass
from .config import CONF
from PIL import Image
import multiprocessing
import os
import sys
import traceback

from .testing_util import run_test


@dataclass
class PromptData:
    """Represents aggregated data for a single prompt."""
    pid: int
    prompt_text: str
    label: int
    features: Dict[str, float]
    avg_base_score: float
    avg_ft_score: float
    image:Image = None

    @classmethod
    def get_all_feature_names(cls) -> List[str]:
        """Returns the complete list of feature names in the correct order."""
        return CONF.base_feature_names + ['avg_base_score', 'avg_ft_score']
        # return CONF.base_feature_names + ['avg_ft_score']

    def to_feature_array(self) -> np.ndarray:
        """Constructs the feature vector in a consistent order."""
        base_features = [self.features[k] for k in CONF.base_feature_names]
        return np.array(base_features + [self.avg_base_score, self.avg_ft_score])
        # return np.array(base_features + [self.avg_ft_score])

def tpr_at_fpr(y_true: np.ndarray, y_score: np.ndarray, target_fpr: float) -> float:
    fpr, tpr, _ = roc_curve(y_true, y_score)
    if not np.all(np.diff(fpr) >= 0):
        sort_idx = np.lexsort((tpr, fpr))
        fpr, tpr = fpr[sort_idx], tpr[sort_idx]
    return float(np.interp(target_fpr, fpr, tpr))

def write_report(path: Path, content: str):
    path.write_text(content)
    print(f"Report saved to {path}")

def compute_score_safe(response: str, label: str, mode='math') -> float:
    if mode == 'math':
        from .compute_score import compute_score
        return compute_score(response, label)
    elif mode == 'code':
        from .code_score import compute_score
        return compute_score(response, label)
    elif mode == 'toolrl':
        from .toolrl_score import compute_score
        return compute_score(response, label)
    elif mode == 'if':
        from .ifmulti import check
        return check(response, label)
    else:
        raise ValueError('No mode selected.')

def clean_infs(X: np.ndarray) -> np.ndarray:
    X = X.astype(float, copy=True)
    inf_mask = ~np.isfinite(X)
    if inf_mask.any():
        X[inf_mask] = np.nan
    return X

def load_prompt_data_from_npz(npz_path: Path) -> list[PromptData]:
    """Reconstructs the list of PromptData objects from a saved .npz file."""
    print(f"Loading cached metrics from {npz_path}...")
    data = np.load(npz_path, allow_pickle=True)
    feats = data['feats']
    labels = data['labels']
    avg_base_scores = data['avg_base_scores']
    avg_ft_scores = data['avg_ft_scores']
    if (feats.shape[1] == 5
            and np.allclose(feats[:, -2], avg_base_scores, equal_nan=True)
            and np.allclose(feats[:, -1], avg_ft_scores, equal_nan=True)):
        raise ValueError("Legacy five-column cache: use scripts/evaluate_cached.py to preserve its feature layout.")
    prompt_texts = data.get('prompt_texts', [f"prompt_{i}" for i in range(len(labels))])

    prompts = []
    for i in range(len(labels)):
        features_dict = {name: feats[i, j] for j, name in enumerate(CONF.base_feature_names)}
        prompt = PromptData(
            pid=i,
            prompt_text=prompt_texts[i],
            label=labels[i],
            features=features_dict,
            avg_base_score=avg_base_scores[i],
            avg_ft_score=avg_ft_scores[i]
        )
        prompts.append(prompt)
    print(f"Successfully loaded {len(prompts)} prompts from cache.")
    return prompts

def export_split_responses(prompts: List[PromptData], out_dir: Path, response_paths: Dict[str, List[Path]]):
    """
    Identifies prompts in 'all-0' and 'all-1' splits and exports their
    base and finetuned model responses to CSV files for comparison.
    """
    print("\nExporting responses for 'all-0' and 'all-1' splits...")

    # Define splits based on scores
    all_zero_flags = np.array([abs(p.avg_base_score) < CONF.score_tolerance and abs(p.avg_ft_score) < CONF.score_tolerance for p in prompts])
    all_one_flags = np.array([abs(p.avg_base_score - 1.0) < CONF.score_tolerance for p in prompts])
    mask_all0 = all_zero_flags
    mask_all1 = all_one_flags & ~mask_all0

    # Read all response files into lookups
    lookups = defaultdict(lambda: defaultdict(list))
    for model_type, paths in response_paths.items():
        for path in paths:
            if path.exists():
                with open(path, 'r') as f:
                    for line in f:
                        record = json.loads(line)
                        lookups[model_type][record['prompt']].append(record['response'])

    # Prepare data for export
    export_data = {'all-0': [], 'all-1': []}
    for i, p in enumerate(prompts):
        if mask_all0[i] or mask_all1[i]:
            split_name = 'all-0' if mask_all0[i] else 'all-1'
            export_data[split_name].append({
                "prompt": p.prompt_text,
                "label": "member" if p.label == 1 else "non-member",
                "avg_ft_score": p.avg_ft_score,
                "avg_base_score": p.avg_base_score,
                "finetuned_responses": lookups['ft'].get(p.prompt_text, ["N/A"]),
                "base_responses": lookups['base'].get(p.prompt_text, ["N/A"]),
            })

    # Write to CSV
    for split_name, data in export_data.items():
        if data:
            df = pd.DataFrame(data)
            out_path = out_dir / f"responses_{split_name}_split.csv"
            df.to_csv(out_path, index=False)
            print(f"Saved {len(df)} responses for '{split_name}' split to {out_path}")
            
            

def _temp_run(sample, generation, debug, result, metadata_list, timeout):
    with open(os.devnull, "w") as devnull:
        sys.stdout = devnull
        sys.stderr = devnull
        try:
            res, metadata = run_test(in_outs=sample, test=generation, debug=debug, timeout=timeout)
            result.append(res)
            metadata_list.append(metadata)
        except Exception:
            # print(e) # some tracebacks are extremely long.
            traceback.print_exc(10)
            result.append([-1 for i in range(len(sample["inputs"]))])
            metadata_list.append({})


def check_correctness(in_outs: Optional[dict], generation, timeout=10, debug=True):
    """Check correctness of code generation with a global timeout.
    The global timeout is to catch some extreme/rare cases not handled by the timeouts
    inside `run_test`"""

    manager = multiprocessing.Manager()
    result = manager.list()
    metadata_list = manager.list()
    p = multiprocessing.Process(target=_temp_run, args=(in_outs, generation, debug, result, metadata_list, timeout))
    p.start()
    p.join(timeout=timeout + 1)
    if p.is_alive():
        p.kill()
        # p.terminate()
    if not result:
        # consider that all tests failed
        result = [[-1 for i in range(len(in_outs["inputs"]))]]
        if debug:
            print("global timeout")
    return result[0], metadata_list
