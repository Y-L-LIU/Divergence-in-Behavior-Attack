# Validation record

Scope: the portable text-only MATH runner and CPU feature-cache evaluation. No paper-quality evaluation was attempted.

## Environment

- Linux, Python 3.11.15
- NVIDIA H800, 80 GB; one previously idle GPU selected explicitly
- PyTorch 2.9.0+cu126; Transformers 4.55.2
- NumPy 1.26.4; pandas 3.0.3; PyArrow 24.0.0
- scikit-learn 1.8.0; SciPy 1.17.1
- Pillow 11.3.0; tqdm 4.67.3; huggingface-hub 0.36.2; safetensors 0.7.0

The inspected environment has no vLLM or qwen-vl-utils installation. The portable text runner does not require either package. Testing used a separate temporary checkout and did not change the research server's source or checkpoint files.

## Checks completed

| Check | Result |
| --- | --- |
| `python audit.py --help` | Passed; argument parsing works without GPU-library imports |
| Python compilation of the runner and `src/` | Passed |
| `python scripts/smoke_test.py`, GPU visibility disabled | Passed |
| Boxed and `####` MATH answers; incorrect/missing answers | Expected scores |
| Missing/duplicate response keys and changed cache manifest | Rejected |
| CPU nested-CV evaluation of 48 temporary synthetic prompts | Completed and produced a finite unified report |
| End-to-end GPU run, 8 member + 8 nonmember prompts | Completed, exit code 0 |
| Four generated response files | 8 records per file; all expected keys present |
| Sample feature file / prompt feature cache | 16 sample records; shape `(16, 7)`, all finite |

The GPU run used the existing Qwen2.5-3B-Instruct reference and the server's merged MATH GRPO checkpoint `iter_0001139_hf`. It read the first 8 rows of each checked-in MATH split, sampled 1 response per prompt, and capped each response at 16 tokens. It exercised generation, verifier scoring, feature calculation, aggregation, and the inner/outer five-fold classifier evaluation.

Equivalent command, with paths supplied by the operator:

```bash
CUDA_VISIBLE_DEVICES=0 python audit.py \
  --finetuned "$FT_MODEL" --reference "$REF_MODEL" \
  --member dataset/math-mia/train_half_test_only.parquet \
  --nonmember dataset/math-mia/test.parquet \
  --out_dir runs/math-smoke \
  --limit 8 --num_samples 1 --max_tokens 16
```

No response caches, weights, or resulting numerical performance reports are included in this change.

## Limits

- The requirements reflect an existing working environment; a clean install remains a release checklist item.
- The full 1,269 + 1,269 prompt audit at 8 responses/prompt and a long generation cap was not run.
- The small, truncated smoke run establishes execution, not model quality or paper-result reproducibility.
- Hugging Face metadata requests failed from both local and server environments. Checkpoint subdirectories, revisions, and their mapping to paper experiments remain unverified. The GPU test used existing local weights.
- The vLLM/VLM, ToolRL, Search-R1, code, instruction-following, and transfer workflows have not been validated by this change.
- Response caches record model paths/IDs, not hashes of weights inside mutable local checkpoint directories.
