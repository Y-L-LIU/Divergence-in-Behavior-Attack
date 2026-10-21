# Divergence-in-Behavior Auditing (DIBA)

Code for **Auditing Data Membership in Reinforcement Learning With Verifiable Rewards** (NDSS 2027).

DIBA compares a model after RL fine-tuning with its reference model. It samples responses to labeled member/nonmember prompts, computes reference-calibrated token statistics and verifier scores, and trains a membership predictor on the resulting prompt-level features.

- [Released checkpoints on Hugging Face](https://huggingface.co/RLMIA)
- [Archived main experiment responses and metrics](artifacts/main/README.md)
- [Repository organization plan](docs/cleanup-plan.md)
- [Validation record](docs/validation.md)

The quick start runs a **text-only MATH audit from existing checkpoints**. Tool Using, Instruction Following, and GLM-4-9B commands are provided below. It does not require RL training, released response caches, or paper result files. The current evaluator runs a unified predictor with five-fold stratified cross-validation; it does not reproduce every experimental setting in the paper.

## 1. Install

Use Linux, Python 3.11, and an NVIDIA CUDA GPU for generation and feature computation. Two model copies must fit on one GPU during feature extraction, with additional memory for full-vocabulary logits. The smoke run passed on an 80 GB H800; a minimum VRAM requirement has not been established.

```bash
git clone https://github.com/Y-L-LIU/Divergence-in-Behavior-Attack.git
cd Divergence-in-Behavior-Attack
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.9.0 --index-url https://download.pytorch.org/whl/cu126
python -m pip install -r requirements.txt
python audit.py --help
```

The commands install PyTorch with CUDA 12.6. Select a compatible PyTorch CUDA wheel if your machine requires a different version.

For CPU-only evaluation of existing features:

```bash
python -m pip install -r requirements-cpu.txt
python scripts/smoke_test.py
```

The smoke script creates temporary synthetic features, exercises answer parsing and cache validation, and runs the classifier pipeline. Its numbers have no experimental meaning.

## 2. Select and download checkpoints

Set `FT_MODEL` to a **merged Hugging Face checkpoint directory**, containing `config.json`, tokenizer files, and weights (plus a shard index when needed). An unmerged LoRA adapter or a distributed training checkpoint is not accepted directly.

Set `REF_MODEL` to the corresponding model **before RL**. For a target trained from Qwen2.5-3B-Instruct, the reference is `Qwen/Qwen2.5-3B-Instruct`. Both models must have the same tokenizer vocabulary, special-token IDs, and chat formatting.

Models are released as separate repositories under the [RLMIA organization](https://huggingface.co/RLMIA). Select an actual model repository ID, such as `RLMIA/GRPO-3B-MATH`.

| MATH checkpoint | Hugging Face repository |
| --- | --- |
| GRPO, 3B | [RLMIA/GRPO-3B-MATH](https://huggingface.co/RLMIA/GRPO-3B-MATH) |
| GRPO, 7B | [RLMIA/GRPO-7B-MATH](https://huggingface.co/RLMIA/GRPO-7B-MATH) |
| DAPO, 3B | [RLMIA/DAPO-3B-MATH](https://huggingface.co/RLMIA/DAPO-3B-MATH) |
| DAPO, 7B | [RLMIA/DAPO-7B-MATH](https://huggingface.co/RLMIA/DAPO-7B-MATH) |

The organization also hosts shadow MATH, Geo3K, instruction-following, Search-R1, and tool-use checkpoints. The [checkpoint inventory](configs/models.json) records repository IDs, pinned revisions, model architectures, and file names. A published checkpoint may require a task-specific runner; the quick start below is for text MATH.

Download the 3B GRPO MATH target and its Qwen2.5-3B-Instruct reference:

```bash
hf download RLMIA/GRPO-3B-MATH \
  --revision a22a23d5fc8d926617b6cefa53d85a4cf5b50fa3 \
  --local-dir checkpoints/GRPO-3B-MATH
hf download Qwen/Qwen2.5-3B-Instruct \
  --revision aa8e72537993ba99e69dfaafa59ed015b17504d1 \
  --local-dir checkpoints/Qwen2.5-3B-Instruct
export FT_MODEL="$PWD/checkpoints/GRPO-3B-MATH"
export REF_MODEL="$PWD/checkpoints/Qwen2.5-3B-Instruct"
```

Both repositories are public and can be downloaded without a token.

Already downloaded models can be used by setting these two variables directly. Keep checkpoint directories immutable during a run. The runner records model paths/IDs but does not hash multi-gigabyte weights.

## 3. Data

These files are already tracked in Git:

| Files | Rows | Use |
| --- | ---: | --- |
| `dataset/math-mia/train_half_test_only.parquet` | 1,269 | Candidate member split for the corresponding MATH target |
| `dataset/math-mia/test.parquet` | 1,269 | Candidate nonmember split |
| `dataset/geo3k/{train,test}.parquet` | 1,000 / 1,002 | Image-based task; separate VLM path |
| `dataset/math-shadow/{train_shadow,test}.parquet` | 1,000 each | Historical shadow data |

Membership labels must agree with the selected target's actual RL training split. A filename alone does not establish membership for an arbitrary checkpoint. The files ending in `1.parquet` under `math-shadow` duplicate their corresponding original splits; they are not independent datasets.

Each Parquet row for `audit.py` contains:

```python
{
    "prompt": [{"role": "user", "content": "Solve ... Output the final answer after ####."}],
    "reward_model": {"ground_truth": "42", "style": "rule"},
}
```

`conversations` is also accepted in place of `prompt`. All chat turns are preserved, and a default system message is added if absent. Rows are assigned positional `row_id` values after loading. The MATH verifier accepts the final `\boxed{...}` answer or the answer following `####`, using the repository's existing string normalization. It is not a symbolic-equivalence verifier.

## 4. Run a small GPU smoke check

Run from the repository root and select an available GPU:

```bash
CUDA_VISIBLE_DEVICES=0 python audit.py \
  --finetuned "$FT_MODEL" \
  --reference "$REF_MODEL" \
  --member dataset/math-mia/train_half_test_only.parquet \
  --nonmember dataset/math-mia/test.parquet \
  --out_dir runs/math-smoke \
  --limit 8 --num_samples 1 --max_tokens 16
```

This checks four response files, token features, aggregation, and nested classifier cross-validation on 8 prompts per class. The short generations test execution only; they are unsuitable for accuracy or membership conclusions. For fewer than 7 prompts per class, add `--skip_eval`.

## 5. Run the complete audit

```bash
CUDA_VISIBLE_DEVICES=0 python audit.py \
  --finetuned "$FT_MODEL" \
  --reference "$REF_MODEL" \
  --member dataset/math-mia/train_half_test_only.parquet \
  --nonmember dataset/math-mia/test.parquet \
  --out_dir runs/math-audit \
  --num_samples 8 --max_tokens 2048 \
  --temperature 0.7 --top_p 0.9 --repetition_penalty 1.05
```

Defaults follow the released audit configuration: 8 responses per prompt, temperature 0.7, top-p 0.9, repetition penalty 1.05, and a 2,048-token generation cap. **Audit sampling and RL training rollouts are different settings.** Select a longer generation cap explicitly if the checkpoint/task requires it; truncated reasoning can affect verifier scores.

The portable runner uses Transformers and generates one response at a time. It prioritizes a small dependency footprint and bounded generation memory. Historical `main.py` uses vLLM and has separate dependencies; vLLM and Transformers outputs need not match bit for bit.

Generation for each model runs in a separate process so its GPU memory is released before the next stage. Feature extraction then loads both models onto one visible GPU. Multi-GPU feature sharding is part of the server-script migration plan.

To separate stages, use the same arguments with `--stage generate`, then `--stage features`. Feature computation rebuilds sample features from validated responses. Add `--skip_eval` to save prompt features without fitting the predictor.

## 6. Outputs and cached evaluation

```text
runs/math-audit/
├── run_config.json
├── base/
│   ├── run_config.json
│   ├── responses_base_member.jsonl
│   └── responses_base_nonmember.jsonl
├── responses_ft_member.jsonl
├── responses_ft_nonmember.jsonl
├── computed_features.jsonl
├── metrics_prompt.npz
├── report_unified_model.txt
└── evaluation_summary.csv
```

Response records contain `prompt`, `response`, `row_id`, and `sample_id`. Features are paired by IDs and averaged per prompt. The seven predictor inputs are five calibrated token features (negative log-likelihood, zlib-normalized likelihood, reverse-KL estimate, Min-K%++ statistic, entropy) and the reference/target mean verifier scores.

The predictor stacks a random forest and logistic regression, with an inner five-fold split and outer five-fold stratified evaluation. At least 7 prompts per class are needed to execute the nested splits; substantially more are needed for meaningful evaluation. Target/reference language models remain fixed throughout classifier cross-validation.

The inherited evaluator also writes `report_specialist_*.txt` and placeholder specialist rows in the CSV. Specialist evaluation is disabled in the current implementation; use the unified report.

Re-evaluate saved prompt features without loading models:

```bash
python audit.py \
  --load_from_npz runs/math-audit/metrics_prompt.npz \
  --out_dir runs/math-audit-reeval
```

Use a fresh output directory when changing models, data, or sampling settings. Manifests check dataset hashes and run options; response validation rejects missing/duplicate sample IDs. Interrupted response writes use `.partial` files and restart that file on the next run. Caches created by the older runner have no matching manifest and should be kept in separate directories.

## 7. Tool Using

Download the released target and use the same Qwen2.5-3B-Instruct reference as above:

```bash
hf download RLMIA/GRPO-3B-TOOL-USING \
  --revision a2cfbc61095a704f9d55e1612c7eb79352926d76 \
  --local-dir checkpoints/GRPO-3B-TOOL-USING
CUDA_VISIBLE_DEVICES=0 python audit.py --task tool \
  --finetuned checkpoints/GRPO-3B-TOOL-USING \
  --reference checkpoints/Qwen2.5-3B-Instruct \
  --member dataset/toolrl/member.parquet \
  --nonmember dataset/toolrl/nonmember_xlam.parquet \
  --out_dir runs/tool-audit
```

The included server split has 1,498 member and 1,500 nonmember rows. Full system messages, tool descriptions, and conversation turns are retained. The verifier scores tool-call names and arguments; it does not execute tools. It returns the server audit's correctness component, excluding the separate format/length training rewards. See [the data notes](dataset/toolrl/README.md).

For an execution check, append `--limit 2 --num_samples 2 --max_tokens 16 --skip_eval`.

## 8. Instruction Following

Both [3B](https://huggingface.co/RLMIA/GRPO-3B-INSTRUCTION-FOLLOWING) and [7B](https://huggingface.co/RLMIA/GRPO-7B-INSTRUCTION-FOLLOWING) targets are released. Use the corresponding Qwen2.5-Instruct reference of the same size.

```bash
python -m pip install -r requirements-instruction.txt
python -m nltk.downloader punkt_tab
hf download RLMIA/GRPO-7B-INSTRUCTION-FOLLOWING \
  --revision b22436d7f3c3c73b0b46183d70993f411d386af5 \
  --local-dir checkpoints/GRPO-7B-INSTRUCTION-FOLLOWING
hf download Qwen/Qwen2.5-7B-Instruct --local-dir checkpoints/Qwen2.5-7B-Instruct
CUDA_VISIBLE_DEVICES=0 python audit.py --task instruction \
  --finetuned checkpoints/GRPO-7B-INSTRUCTION-FOLLOWING \
  --reference checkpoints/Qwen2.5-7B-Instruct \
  --member /path/to/instruction/member.parquet \
  --nonmember /path/to/instruction/nonmember.parquet \
  --out_dir runs/instruction-audit
```

Original member/nonmember split files are not yet included. Supply the matching MulDimIF splits. Each row needs `prompt` (or `conversations`) and a nonempty `constraints` list; each constraint has category, type, and rubric text, for example:

```python
{
    "prompt": [{"role": "user", "content": "Write a greeting in ALL UPPERCASE."}],
    "constraints": [["Language", "English: All Uppercase", "Must use all uppercase letters."]],
}
```

The verifier averages the per-constraint checks. It supports the repository's existing content, length, language, punctuation, and format rules without an external LLM judge.

For a synthetic execution fixture, run `python scripts/make_instruction_smoke.py --out_dir runs/instruction-fixture` and use its member/nonmember files with `--skip_eval`. Those synthetic labels are not actual training memberships.

## 9. GLM-4-9B MATH

**Checkpoint publication is in progress.** The commands below require the complete weights in the linked model repository; the repository's existence alone does not confirm availability.

The GLM model-family extension uses the same MATH data interface:

```bash
hf download RLMIA/GRPO-GLM4-9B-MATH --local-dir checkpoints/GRPO-GLM4-9B-MATH
hf download zai-org/GLM-4-9B-0414 \
  --revision 645b8482494e31b6b752272bf7f7f273ef0f3caf \
  --local-dir checkpoints/GLM-4-9B-0414
CUDA_VISIBLE_DEVICES=0 python audit.py --task math \
  --finetuned checkpoints/GRPO-GLM4-9B-MATH \
  --reference checkpoints/GLM-4-9B-0414 \
  --member dataset/math-mia/train_half_test_only.parquet \
  --nonmember dataset/math-mia/test.parquet \
  --out_dir runs/glm-math-audit
```

This is the later 60-epoch matched-setting checkpoint, iteration 1,139, with an 8,192-token training response cap. The earlier iteration-779 checkpoint is a separate run. See the released model card for the recorded training configuration. As with other targets, both model copies must fit on the selected GPU during feature extraction.

## 10. Main experiment archives

[artifacts/main](artifacts/main/README.md) contains the original four MATH runs' metrics and reports. Generated target responses and sample features are available from [RLMIA/DIBA-MATH-RESULTS](https://huggingface.co/datasets/RLMIA/DIBA-MATH-RESULTS).

These archived runs use temperature 0.5 and a five-column detector feature matrix. Use `scripts/evaluate_cached.py` for them; the current runner's seven-column loader rejects that legacy layout. The archive notes describe the preserved non-finite values and missing reference-response files.

## Scope of the current release

| Component | Status |
| --- | --- |
| MATH / Tool Using / GLM generation and features | Portable text runner; small GPU checks recorded |
| Instruction Following | Runner, verifier and synthetic checks; original split files pending |
| Current and archived feature-cache evaluation | Separate CPU entry points for the two feature layouts |
| Four main MATH experiments | Original metrics, reports, target responses and sample features |
| Historical vLLM/VLM path (`main.py`) | Retained; outside the portable quick start |
| Search-R1, training, transfer and analysis utilities | Separate workflows tracked in the organization plan |
