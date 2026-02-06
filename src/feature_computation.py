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
from vllm import LLM, SamplingParams
from qwen_vl_utils import fetch_image
import pandas as pd
from .utils import PromptData, compute_score_safe
from .config import CONF
from PIL import Image
import io, base64
from typing import Union

def encode_image(img):
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    img_bytes = buffer.getvalue()
    img_base64 = base64.b64encode(img_bytes).decode("utf-8")
    return img_base64    

def _row_to_messages(
    text,
    image_ref=None
):
    """Build Qwen-style messages (official schema). Single image is optional."""
    msgs = []
    if image_ref:
        msgs.append({"role": "system", "content": [{"type": "text", "text": "You are a helpful assistant."}]})
        img_bytes = image_ref[0]['bytes']
        image = Image.open(io.BytesIO(img_bytes))
        image = fetch_image({'image':image})
        msgs.append({
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": text},
            ],})
    else:
        msgs.append({"role": "system", "content":  "You are a helpful assistant."})
        msgs.append({"role": "user", "content": text})
        image=None
    print(msgs)
    return msgs, image



class NoisyLogitsProcessor:
    def __init__(self, noise_scale: float = 0.1):
        self.noise_scale = noise_scale

    def __call__(
        self,
        past_tokens_ids: Union[list[int], tuple[int]],
        logits: torch.Tensor,
    ) -> torch.Tensor:
        # Add noise to logits; ignore past_tokens_ids as it's not needed for this processor
        noise = torch.randn_like(logits) * self.noise_scale
        return logits + noise

def generate_responses(
    model_path: str,
    df_path,
    outfile: Path,
    temp=CONF.temperature,
    noise=0.0
) -> None:
    """
    Batch-generate with vLLM using Qwen's chat template (image optional per row).
    DataFrame expectations:
      - text_col: str question/prompt
      - image_col: str or Path to a single image (URL or local path) OR NaN/None for text-only

    Output JSONL fields per sample:
      - prompt, response, sample_id, row_id, has_image
    """
    df = pd.read_parquet(df_path)
    # df = df[:10]
    if outfile.exists():
        print(f"[cached] Skipping generation for {outfile.name}")
        return

    # --- tokenizer / processor just to build the official chat template ---
    if 'vl' not in model_path.lower():
        processor = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    else:
        processor = AutoProcessor.from_pretrained(model_path,trust_remote_code=True)
    # (Qwen says to use apply_chat_template for correct prompt wiring.) :contentReference[oaicite:1]{index=1}

    # --- vLLM engine (single-image per prompt; text-only OK) ---

    # --- build per-row requests (always using the Qwen chat template) ---
    requests: List[Dict[str, Any]] = []
    metas: List[Dict[str, Any]] = []

    for ridx, row in df.iterrows():
        if 'prompt' in row.keys():
            msg, image =  _row_to_messages(row['prompt'][0]['content'], row.get('images',''))
        else:
            msg, image =  _row_to_messages(row['conversations'][0]['content'], row.get('images',''))

        # Convert messages -> chat prompt string (Qwen official way)
        chat_text = processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
        # If there is an image, let qwen_vl_utils locate/prepare it; otherwise no mm data.
        # print(chat_text,image)
            # Single-image guaranteed: take the first image item
        if image:
            mm_data = {"image": [image]}
            req = {"prompt": chat_text,"multi_modal_data":mm_data}
        else:
            req = {"prompt": chat_text}

        requests.append(req)
        metas.append({"row_id": ridx,})
        
    llm = LLM(
        model=model_path,
        tensor_parallel_size=max(1, torch.cuda.device_count()),
        dtype="auto",
        gpu_memory_utilization=0.9,
        trust_remote_code=True,
        limit_mm_per_prompt={"image": 1} if 'vl'in model_path.lower() else None,   # we only allow 1 image per request
    )
    sampling_params = SamplingParams(
        n=CONF.num_samples,
        max_tokens=CONF.max_tokens,
        temperature=temp,
        top_p=CONF.top_p,
        repetition_penalty=CONF.repetition_penalty,
        logits_processors=[NoisyLogitsProcessor(noise_scale=noise)] if noise else None
    )

    print(f"Generating responses for {len(requests)} rows -> {outfile.name}")
    outfile.parent.mkdir(parents=True, exist_ok=True)

    with outfile.open("w", encoding="utf-8") as f:
        # batch over requests
        for i in tqdm(range(0, len(requests), CONF.vllm_batch_size),
                      desc=f"Generating {outfile.stem}"):
            batch_reqs = requests[i:i + CONF.vllm_batch_size]
            batch_meta = metas[i:i + CONF.vllm_batch_size]

            outputs = llm.generate(batch_reqs, sampling_params)
            for req, out, meta in zip(batch_reqs, outputs, batch_meta):
                for sample in out.outputs:
                    rec = {
                        "prompt": req["prompt"],               # Qwen chat-formatted prompt
                        "response": sample.text.strip(),
                        "sample_id": sample.index,
                        "row_id": meta["row_id"],
                    }
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    
# def generate_responses(model_path: str, prompts_json: Path, outfile: Path):
#     """Generates and saves model responses to a JSONL file."""
#     if outfile.exists():
#         print(f"[cached] Skipping generation for {outfile.name}")
#         return
#     print(f"Generating responses for {prompts_json.name} -> {outfile.name}")
#     tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
#     if tokenizer.pad_token_id is None: tokenizer.pad_token = tokenizer.eos_token
#     with open(prompts_json, 'r') as f:
#         prompts = [item["input"] for item in json.load(f)]
    
#     chat_prompts = [build_chat_prompt(tokenizer, q) for q in prompts]
    
#     # Associate original query with chat prompt for later retrieval
#     prompt_map = {chat_p: orig_p for chat_p, orig_p in zip(chat_prompts, prompts)}
    
#     llm = LLM(model_path, tensor_parallel_size=max(1, torch.cuda.device_count()), dtype="auto", gpu_memory_utilization=0.9, trust_remote_code=True)
#     sampling_params = SamplingParams(n=CONF.num_samples, max_tokens=CONF.max_tokens, temperature=CONF.temperature, top_p=CONF.top_p, repetition_penalty=CONF.repetition_penalty)
    
#     with outfile.open("w") as f:
#         for i in tqdm(range(0, len(chat_prompts), CONF.vllm_batch_size), desc=f"Generating {outfile.stem}"):
#             batch_chat_prompts = chat_prompts[i:i + CONF.vllm_batch_size]
#             outputs = llm.generate(batch_chat_prompts, sampling_params)
#             for request_output in outputs:
#                 original_prompt_text = prompt_map[request_output.prompt]
#                 for sample in request_output.outputs:
#                     record = {
#                         "prompt": request_output.prompt, # Store original prompt
#                         "response": sample.text.strip(),
#                         "sample_id": sample.index
#                     }
#                     f.write(json.dumps(record, ensure_ascii=False) + "\n")
#     return
def _decode_base64_image(b64) -> Image.Image:
    if isinstance(b64, str) and b64.startswith("data:image"):
        b64 = b64.split(",", 1)[1]
    if isinstance(b64, (bytes, bytearray)):
        b64 = base64.b64encode(b64).decode()
    img = Image.open(io.BytesIO(base64.b64decode(b64, validate=False))).convert("RGB")
    return img


def _compute_token_metrics(logits: torch.Tensor, token_ids: torch.Tensor, temperature: float = 0.7) -> Dict[str, Any]:
    """Helper to compute metrics from a single model's logits."""
    # Apply temperature scaling to logits before softmax
    temp_logits = logits / temperature
    logp = F.log_softmax(temp_logits, dim=-1)
    probs = logp.exp()

    # Log-probability of the actual generated tokens
    token_lp_actual = torch.gather(logp, -1, token_ids.unsqueeze(-1)).squeeze(-1)

    # Negative Log-Likelihood
    neg_ll = -token_lp_actual.mean().item()

    # Entropy (temperature-aware)
    entropy = -(probs * logp).sum(-1).mean().item()

    # Z-score and Minkowski Score
    mu = (probs * logp).sum(-1)
    var = (probs * logp.pow(2)).sum(-1) - mu.pow(2)
    z_scores = (token_lp_actual - mu) / var.clamp_min(1e-12).sqrt()
    k = max(1, int(z_scores.numel() * 0.2))
    minkowski_score = z_scores.topk(k, largest=False).values.mean().item()

    return {
        "neg_ll": neg_ll,
        "entropy": entropy,
        "minkowski": minkowski_score,
        "token_logp_scalar": logp,
        "token_logp_actual": token_lp_actual,
    }


def _build_inputs(prompt: str, response: str, image, tok, device):
    """
    Returns: inputs_full(dict on device), full_ids(Tensor), prompt_len(int)
    Works for both text-only tokenizers and Qwen-VL AutoProcessor.
    """
    if image:
        if not prompt.endswith("\n"):
            prompt = prompt + "\n"
        full_chat = f"{prompt}{response}"

        inputs_prompt = tok(text=[prompt], images=[image], return_tensors="pt")
        inputs_full   = tok(text=[full_chat], images=[image], return_tensors="pt")

        inputs_prompt = {k: v.to(device) for k, v in inputs_prompt.items()}
        inputs_full   = {k: v.to(device)   for k, v in inputs_full.items()}
        full_ids      = inputs_full["input_ids"]
        prompt_len    = int(inputs_prompt["input_ids"].shape[1])
        return inputs_full, full_ids, prompt_len
    else:
        prompt_ids = tok(prompt, return_tensors="pt").input_ids.to(device)
        resp_ids   = tok(response, return_tensors="pt").input_ids.to(device)
        full_ids   = torch.cat([prompt_ids, resp_ids], dim=1)
        attn_mask  = torch.ones_like(full_ids)
        inputs_full = {"input_ids": full_ids, "attention_mask": attn_mask}
        return inputs_full, full_ids, int(prompt_ids.shape[1])


@torch.inference_mode()
def compute_mia_features(prompt: str, response: str, img, model_ft, model_ref, tok, device) -> Dict[str, float]:
    """Computes MIA features, now with calibration against a reference model."""

    inputs_full, full_ids, prompt_len = _build_inputs(prompt, response, img, tok, device)

    # Build labels: mask prompt tokens with -100
    labels = full_ids.clone()
    labels[:, :prompt_len] = -100

    # Response positions (shifted by one for next-token prediction)
    resp_mask = (labels[:, 1:] != -100)
    token_ids_resp = full_ids[:, 1:][resp_mask]

    # Forward both models; works for both LM and VLM since inputs_full is a dict
    out_ft  = model_ft(**inputs_full, labels=labels)
    out_ref = model_ref(**inputs_full, labels=labels)

    logits_ft  = out_ft.logits[:, :-1, :][resp_mask]
    logits_ref = out_ref.logits[:, :-1, :][resp_mask]

    # Compute base metrics for both models
    metrics_ft = _compute_token_metrics(logits_ft, token_ids_resp, CONF.temperature)
    metrics_ref = _compute_token_metrics(logits_ref, token_ids_resp, 0.7)  # Reference model uses T=1.0

    # --- Calibrated Metrics (Fine-tuned - Reference) ---
    neg_ll_diff = metrics_ft["neg_ll"] - metrics_ref["neg_ll"]
    entropy_diff = metrics_ft["entropy"] - metrics_ref["entropy"]
    minkowski_diff = metrics_ft["minkowski"] - metrics_ref["minkowski"]

    # zlib ratio is based on the calibrated log-likelihood
    zlib_len = len(zlib.compress(response.encode()))
    zlib_entropy_ratio = neg_ll_diff / zlib_len if zlib_len > 0 else 0.0

    # --- Reference-aware Metrics (calculated using both models) ---
    lp_ft_actual = metrics_ft["token_logp_actual"]
    lp_ref_actual = metrics_ref["token_logp_actual"]
    rev_kl_div = (torch.exp(lp_ref_actual - lp_ft_actual) - (lp_ref_actual - lp_ft_actual) - 1).mean().item()
    return {
        "neg_log_likelihood": neg_ll_diff,
        "zlib_entropy_ratio": zlib_entropy_ratio,
        "minkowski_0.2": minkowski_diff,
        "reverse_kl_div": rev_kl_div,
        "avg_entropy": entropy_diff,
    }

def get_imgs(df):
    imgs = []
    if 'images' not in df.columns:
        return [None for _ in range(len(df))]
    for idx, row in df.iterrows():
        image_ref = row.get('images', '')
        img_bytes = image_ref[0]['bytes']
        image = Image.open(io.BytesIO(img_bytes))
        image = fetch_image({'image':image})
        imgs.append(image)
    return imgs

def process_and_compute_features(args, models, tokenizers) -> List[PromptData]:
    """Orchestrates feature computation with improved caching."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir)
    base_dir = Path(args.base_response)
    cache_path = out_dir / "computed_features.jsonl"
    
    # Define all required columns for the cache
    required_cols = set(CONF.base_feature_names + ["finetune_score", "base_score", "prompt", "sample_id", "group"])

    # --- Caching Logic ---
    use_cache = False
    if cache_path.exists():
        with open(cache_path, 'r') as f:
            try:
                first_line = f.readline()
                if first_line:
                    cached_cols = set(json.loads(first_line).keys())
                    if required_cols.issubset(cached_cols):
                        print(f"All required features found in cache file: {cache_path}")
                        use_cache = True
                    else:
                        missing = required_cols - cached_cols
                        print(f"Cache is missing features: {missing}. Re-computing all features.")
                else: # Cache is empty
                    print("Cache file is empty. Re-computing all features.")
            except (json.JSONDecodeError, IndexError):
                print("Could not parse cache file. Re-computing all features.")
    else:
        print("No cache file found. Computing all features.")
        
    if use_cache:
        print("Loading sample-level features from cache...")
        with open(cache_path, 'r') as f:
            all_samples_data = [json.loads(line) for line in f]
    else:
        print("Computing sample-level features...")
        m_df = pd.read_parquet(args.member)
        nm_df = pd.read_parquet(args.nonmember)
        if 'reward_model' in m_df.keys():
            labels_m = [x['ground_truth'] for x in m_df['reward_model']]
            labels_nm = [x['ground_truth'] for x in nm_df['reward_model']]
        else:
            labels_m = [x for x in m_df['constraints']]
            labels_nm = [x for x in nm_df['constraints']]

        # with open(args.label_m) as f: labels_m = [item["label"] for item in json.load(f)]
        # with open(args.label_nm) as f: labels_nm = [item["label"] for item in json.load(f)]
        img_m = get_imgs(m_df)
        img_nm = get_imgs(nm_df)
        paths = {
            "member": {"ft": out_dir/args.out_member, "base": base_dir/"responses_base_member.jsonl", "labels": labels_m, 'images': img_m}, 
            "nonmember": {"ft": out_dir/args.out_nonmember, "base": base_dir/"responses_base_nonmember.jsonl", "labels": labels_nm, 'images': img_nm}
        }
        all_samples_data = []
        mode = 'math' #!!!
        print(f'using {mode} mode')
        for group, data in paths.items():
            with open(data["ft"]) as f: ft_rows = [json.loads(line) for line in f]
            with open(data["base"]) as f: base_rows = [json.loads(line) for line in f]
            sample_labels = [lbl for lbl in data["labels"] for _ in range(CONF.num_samples)]
            sample_images = [img for img in data["images"] for _ in range(CONF.num_samples)] 
            for r_ft, r_base, lbl, img in tqdm(zip(ft_rows, base_rows, sample_labels, sample_images), total=len(ft_rows), desc=f"Processing {group}"):
                if img:
                    features = compute_mia_features(r_ft["prompt"], r_ft["response"], img, models["ft"], models["ref"], tokenizers["ft"], device)
                else:
                    features = compute_mia_features(r_ft["prompt"], r_ft["response"], None, models["ft"], models["ref"], tokenizers["ft"], device)
                features["finetune_score"] = compute_score_safe(r_ft["response"], lbl, mode)
                features["base_score"] = compute_score_safe(r_base["response"], lbl, mode)
                features["prompt"] = r_ft["prompt"]
                features["sample_id"] = r_ft["sample_id"]
                features["group"] = group
                all_samples_data.append(features)

        # Save the newly computed features to cache
        print(f"Saving computed features to cache at {cache_path}...")
        cache_path.parent.mkdir(exist_ok=True)
        with open(cache_path, 'w') as f:
            for item in all_samples_data:
                f.write(json.dumps(item) + '\n')
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
    return aggregated_prompts