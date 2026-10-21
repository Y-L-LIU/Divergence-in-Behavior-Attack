"""Portable text-only MATH audit using the released feature definitions."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--finetuned", help="Merged Hugging Face target checkpoint directory or model ID")
    parser.add_argument("--reference", help="Matching pre-RL checkpoint; must share the target tokenizer")
    parser.add_argument("--member", type=Path)
    parser.add_argument("--nonmember", type=Path)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--base_response", type=Path)
    parser.add_argument("--load_from_npz", type=Path, help="Evaluate existing prompt features on CPU")
    parser.add_argument("--stage", choices=["all", "generate", "features"], default="all")
    parser.add_argument("--role", choices=["base", "ft"], help=argparse.SUPPRESS)
    parser.add_argument("--num_samples", type=int, default=8)
    parser.add_argument("--max_tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--repetition_penalty", type=float, default=1.05)
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--limit", type=int, help="First N rows per split, for smoke checks only")
    parser.add_argument("--skip_eval", action="store_true", help="Save features without classifier CV")
    args = parser.parse_args()
    if not args.load_from_npz and not all([args.finetuned, args.reference, args.member, args.nonmember]):
        parser.error("A full run requires --finetuned, --reference, --member and --nonmember.")
    if args.num_samples < 1 or args.max_tokens < 1 or args.temperature <= 0:
        parser.error("num_samples, max_tokens and temperature must be positive.")
    if not 0 < args.top_p <= 1 or args.repetition_penalty <= 0:
        parser.error("Require 0 < top_p <= 1 and repetition_penalty > 0.")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive.")
    args.base_response = args.base_response or args.out_dir / "base"
    if args.base_response.resolve() == args.out_dir.resolve():
        parser.error("--base_response and --out_dir must be different directories.")
    return args


def load_data(path, limit):
    import pandas as pd

    df = pd.read_parquet(path)
    if limit is not None:
        df = df.iloc[:limit]
    df = df.reset_index(drop=True)
    if df.empty:
        raise ValueError(f"Empty data: {path}")
    key = "prompt" if "prompt" in df else "conversations"
    if key not in df or "reward_model" not in df:
        raise ValueError(f"{path}: need prompt/conversations and reward_model.ground_truth")
    if "images" in df:
        raise ValueError("audit.py supports text-only data; Geo3K needs the separate VLM workflow.")
    for row_id, row in df.iterrows():
        messages = row[key]
        messages = messages.tolist() if hasattr(messages, "tolist") else messages
        if not isinstance(messages, list) or not messages:
            raise ValueError(f"{path}, row {row_id}: expected a nonempty list of chat messages")
        if not all(isinstance(m, dict) and isinstance(m.get("content"), str) and m.get("role") for m in messages):
            raise ValueError(f"{path}, row {row_id}: expected role/content text messages")
        if not isinstance(row["reward_model"], dict) or "ground_truth" not in row["reward_model"]:
            raise ValueError(f"{path}, row {row_id}: missing ground_truth")
    return df


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def ensure_manifest(directory, manifest):
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "run_config.json"
    if path.exists():
        if json.loads(path.read_text()) != manifest:
            raise ValueError(f"Configuration changed in {directory}. Use a fresh output directory.")
    elif any(directory.glob("*.jsonl")) or any(directory.glob("*.npz")):
        raise ValueError(f"Unversioned cache in {directory}. Use a fresh output directory.")
    path.write_text(json.dumps(manifest, indent=2) + "\n")


def prepare(args):
    data = {group: load_data(getattr(args, group), args.limit) for group in ("member", "nonmember")}
    manifest = {
        "schema": 1, "backend": "transformers", "score_mode": "math-boxed-or-hash",
        "reference": args.reference,
        "data_sha256": {group: file_hash(getattr(args, group)) for group in data},
        "num_samples": args.num_samples, "max_tokens": args.max_tokens,
        "temperature": args.temperature, "top_p": args.top_p,
        "repetition_penalty": args.repetition_penalty, "seed": args.seed, "limit": args.limit,
    }
    ensure_manifest(args.base_response, manifest)
    ensure_manifest(args.out_dir, dict(manifest, finetuned=args.finetuned))
    return data


def response_path(args, role, group):
    directory = args.base_response if role == "base" else args.out_dir
    return directory / f"responses_{role}_{group}.jsonl"


def read_responses(path, n_rows, num_samples):
    records = {}
    with path.open() as stream:
        for line in stream:
            row = json.loads(line)
            key = (int(row["row_id"]), int(row["sample_id"]))
            if key in records:
                raise ValueError(f"Duplicate response key {key} in {path}")
            if not isinstance(row.get("prompt"), str) or not isinstance(row.get("response"), str):
                raise ValueError(f"Invalid response record in {path}")
            records[key] = row
    expected = {(r, s) for r in range(n_rows) for s in range(num_samples)}
    if set(records) != expected:
        raise ValueError(f"Incomplete or stale response cache: {path}; use a fresh run directory.")
    return records


def generate(args, data):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

    model_path = args.reference if args.role == "base" else args.finetuned
    pending = []
    for group, df in data.items():
        path = response_path(args, args.role, group)
        if path.exists():
            read_responses(path, len(df), args.num_samples)
            print(f"[cached] {path}", flush=True)
        else:
            pending.append(group)
    if not pending:
        return
    if not torch.cuda.is_available():
        raise RuntimeError("Generation requires a CUDA GPU. Use --load_from_npz for CPU evaluation.")
    set_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.float16, trust_remote_code=True
    ).to("cuda").eval()
    for group in pending:
        df = data[group]
        path = response_path(args, args.role, group)
        temporary = path.with_suffix(".jsonl.partial")
        with temporary.open("w", encoding="utf-8") as stream:
            for row_id, row in df.iterrows():
                messages = row["prompt"] if "prompt" in row else row["conversations"]
                messages = messages.tolist() if hasattr(messages, "tolist") else messages
                # Retain the released MATH system prompt and all conversation turns.
                if messages[0]["role"] != "system":
                    messages = [{"role": "system", "content": "You are a helpful assistant."}] + messages
                prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to("cuda")
                for sample_id in range(args.num_samples):
                    # Per-record seeds make a regenerated file independent of skipped files.
                    set_seed(args.seed + (0 if group == "member" else 1000000) + row_id * args.num_samples + sample_id)
                    with torch.inference_mode():
                        ids = model.generate(
                            **inputs, do_sample=True, temperature=args.temperature, top_p=args.top_p,
                            top_k=0, repetition_penalty=args.repetition_penalty,
                            max_new_tokens=args.max_tokens, pad_token_id=tokenizer.eos_token_id,
                        )
                    response = tokenizer.decode(ids[0, inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
                    if not response:
                        raise ValueError(f"Empty model response at {group}/{row_id}/{sample_id}")
                    record = {"prompt": prompt, "response": response, "row_id": int(row_id), "sample_id": sample_id}
                    stream.write(json.dumps(record, ensure_ascii=False) + "\n")
                print(f"[{args.role}/{group}] {row_id + 1}/{len(df)} prompts", flush=True)
        temporary.replace(path)


def compute_features(args, data):
    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from src.feature_computation import compute_mia_features
    from src.utils import PromptData, compute_score_safe

    # Validate all four caches before loading either checkpoint.
    responses = {
        (role, group): read_responses(response_path(args, role, group), len(df), args.num_samples)
        for group, df in data.items() for role in ("base", "ft")
    }
    tokenizer = AutoTokenizer.from_pretrained(args.finetuned, trust_remote_code=True)
    ref_tokenizer = AutoTokenizer.from_pretrained(args.reference, trust_remote_code=True)
    if tokenizer.get_vocab() != ref_tokenizer.get_vocab() or tokenizer.all_special_ids != ref_tokenizer.all_special_ids:
        raise ValueError("Target/reference must use the same tokenizer vocabulary and special token IDs.")
    if not torch.cuda.is_available():
        raise RuntimeError("Feature computation requires a CUDA GPU.")
    device = torch.device("cuda")
    model_ft = AutoModelForCausalLM.from_pretrained(
        args.finetuned, torch_dtype=torch.float16, trust_remote_code=True
    ).to(device).eval()
    model_ref = AutoModelForCausalLM.from_pretrained(
        args.reference, torch_dtype=torch.float16, trust_remote_code=True
    ).to(device).eval()
    sample_path = args.out_dir / "computed_features.jsonl"
    temporary = sample_path.with_suffix(".jsonl.partial")
    prompts = []
    with temporary.open("w", encoding="utf-8") as stream:
        for group, df in data.items():
            for row_id, row in df.iterrows():
                samples = []
                for sample_id in range(args.num_samples):
                    key = (int(row_id), sample_id)
                    ft = responses[("ft", group)][key]
                    base = responses[("base", group)][key]
                    if ft["prompt"] != base["prompt"]:
                        raise ValueError(f"Target/reference prompt templates differ at {group}/{row_id}")
                    features = compute_mia_features(
                        ft["prompt"], ft["response"], None, model_ft, model_ref, tokenizer, device,
                        temperature=args.temperature,
                    )
                    if not all(np.isfinite(value) for value in features.values()):
                        raise ValueError(f"Non-finite features at {group}/{row_id}/{sample_id}")
                    truth = row["reward_model"]["ground_truth"]
                    record = dict(
                        features, finetune_score=compute_score_safe(ft["response"], truth, "math"),
                        base_score=compute_score_safe(base["response"], truth, "math"),
                        prompt=ft["prompt"], row_id=int(row_id), sample_id=sample_id, group=group,
                    )
                    stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
                    samples.append(record)
                prompts.append(PromptData(
                    pid=len(prompts), prompt_text=samples[0]["prompt"], label=int(group == "member"),
                    features={name: float(np.mean([sample[name] for sample in samples])) for name in features},
                    avg_base_score=float(np.mean([sample["base_score"] for sample in samples])),
                    avg_ft_score=float(np.mean([sample["finetune_score"] for sample in samples])),
                ))
                print(f"[features/{group}] {row_id + 1}/{len(df)} prompts", flush=True)
    temporary.replace(sample_path)
    np.savez(
        args.out_dir / "metrics_prompt.npz",
        feats=np.array([p.to_feature_array() for p in prompts]),
        labels=np.array([p.label for p in prompts]),
        avg_base_scores=np.array([p.avg_base_score for p in prompts]),
        avg_ft_scores=np.array([p.avg_ft_score for p in prompts]),
        prompt_texts=np.array([p.prompt_text for p in prompts]),
    )
    return prompts


def main():
    args = parse_args()
    if args.load_from_npz:
        from src.utils import load_prompt_data_from_npz
        from src.evaluation import run_evaluation
        args.out_dir.mkdir(parents=True, exist_ok=True)
        run_evaluation(load_prompt_data_from_npz(args.load_from_npz), args.out_dir)
        return
    data = prepare(args)
    if args.role:
        generate(args, data)
        return
    if args.stage != "generate" and not args.skip_eval and min(len(df) for df in data.values()) < 7:
        raise ValueError("Use at least 7 rows per class for nested CV, or set --skip_eval.")
    if args.stage in ("all", "generate"):
        for role in ("base", "ft"):
            # Separate processes release model memory before the next generation/feature stage.
            subprocess.run([sys.executable, str(Path(__file__).resolve()), *sys.argv[1:], "--role", role], check=True)
    if args.stage in ("all", "features"):
        prompts = compute_features(args, data)
        if not args.skip_eval:
            from src.evaluation import run_evaluation
            run_evaluation(prompts, args.out_dir)


if __name__ == "__main__":
    main()
