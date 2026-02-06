# feature_computation.py
from __future__ import annotations

import gc
import json
import zlib
from pathlib import Path
from typing import Dict, List, Any
import numpy as np
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor
import pandas as pd
from src.utils import PromptData, compute_score_safe
from src.config import CONF
from PIL import Image
import io, base64
from typing import Union
out_dir='math_dapo_7b_temp/t=0.5'
cache_path = f'{out_dir}/computed_features.jsonl'
with open(cache_path, 'r') as f:
    all_samples_data = [json.loads(line) for line in f]
    print(len(all_samples_data))

# --- Aggregation Logic ---
print("Aggregating sample-level features to prompt-level...")
import numpy as np
from math import ceil

aggregated_prompts = []
all_feature_keys = CONF.base_feature_names

# Group samples into chunks of 8
n_groups = ceil(len(all_samples_data) / CONF.num_samples)
for i in range(n_groups):
    start_idx = i * CONF.num_samples
    end_idx = min((i + 1) * CONF.num_samples, len(all_samples_data))
    group_samples = all_samples_data[start_idx:end_idx]

    # Decide group label — here we use majority label; you can adjust logic
    group_labels = [1 if s["group"] == "member" else 0 for s in group_samples]
    # Majority vote (if tie, pick first or raise error — here we pick first in tie)
    label = max(set(group_labels), key=group_labels.count)

    # Compute average features
    avg_features = {
        key: np.nanmean([
            s.get(key, float('nan')) for s in group_samples
        ]) for key in all_feature_keys
    }

    # Compute average base_score and finetune_score
    # Handle cases where score might be list or scalar
    def safe_get_score(sample, key):
        val = sample.get(key, float('nan'))
        if isinstance(val, list):
            return val[0] if len(val) > 0 else float('nan')
        return val

    avg_base_score = np.nanmean([
        safe_get_score(s, "base_score") for s in group_samples
    ])
    avg_ft_score = np.nanmean([
        safe_get_score(s, "finetune_score") for s in group_samples
    ])

    # Create aggregated PromptData object
    aggregated_prompts.append(PromptData(
        pid=i,
        prompt_text=f"Group_{i}",  # Or you can concatenate/summarize actual prompts if needed
        label=label,
        features=avg_features,
        avg_base_score=avg_base_score,
        avg_ft_score=avg_ft_score
    ))
prompt_data = aggregated_prompts
metrics_path = f'{out_dir}/metrics_prompt.npz'
np.savez(
    metrics_path,
    feats=np.array([p.to_feature_array() for p in prompt_data]),
    labels=np.array([p.label for p in prompt_data]),
    avg_base_scores=np.array([p.avg_base_score for p in prompt_data]),
    avg_ft_scores=np.array([p.avg_ft_score for p in prompt_data]),
    prompt_texts=np.array([p.prompt_text for p in prompt_data])
)
print(f"\nAggregated prompt-level metrics saved to {metrics_path}")
