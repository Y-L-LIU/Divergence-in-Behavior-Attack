# Validation record

Scope: portable text-only auditing, task verifiers, current/archived feature caches, and release artifacts. GPU checks are execution tests, not new paper-quality evaluations.

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
| RLMIA model inventory | Initial 18 public repositories; revision, root config and file list read for each |
| Published GLM checkpoint | Revision `ca52af2f59949c57a33641d6f1028102e6c297b9`; all 16 release files match local sizes and SHA-256, including six weight shards |
| Tool Using GPU run | 2 member + 2 nonmember prompts, 2 samples each; shape `(4, 7)`, all finite |
| GLM-4-9B GPU run | Final iteration-1,139 checkpoint; 2 prompts/class, shape `(4, 7)`, all finite |
| Instruction Following verifier and schema | Uppercase positive/negative examples, empty-rubric rejection, task-schema checks passed |
| Archived five-column feature cache | Layout guard and 16-prompt nested-CV execution check passed |
| Four archived main runs | All 8 target-response files have 10,152 unique `(row_id, sample_id)` records; hashes recorded |

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

The four requested original main MATH experiments are published separately from these smoke runs. Their metrics/reports are in `artifacts/main`, and compressed target responses/sample features are in the accompanying HF dataset. The smoke-run outputs remain temporary.

## Limits

- The requirements reflect an existing working environment; a clean install remains a release checklist item.
- The full 1,269 + 1,269 prompt audit at 8 responses/prompt and a long generation cap was not run.
- The small, truncated smoke run establishes execution, not model quality or paper-result reproducibility.
- The initial 18-model inventory was retrieved through the Hugging Face mirror API and revision-pinned config downloads; user-facing download commands use the official Hugging Face endpoint. Full weights were not downloaded. Reference/training-split mappings for those initial 18 models remain to be confirmed. The newly published GLM checkpoint has a verified reference and member-split mapping; its public file hashes match the local checkpoint used for the GPU check. The GPU test used existing local weights, not a newly downloaded published model.
- Full original Instruction Following splits are not available yet; its verifier/schema checks used synthetic constraints. No claim of reproducing its original experiment is made.
- The vLLM/VLM, Search-R1, code, and transfer workflows remain outside these execution checks.
- Response caches record model paths/IDs, not hashes of weights inside mutable local checkpoint directories.

## Extension details

The Tool Using check used the server merged checkpoint 1,380 and the saved XLAM split. Its tokenizer stores a list under `extra_special_tokens`; the loader maps that list to `additional_special_tokens` for Transformers 4.x without changing the checkpoint files. GLM used the later 60-epoch, 8,192-token-response run and its matching GLM-4-9B-0414 reference.

The four main archive files are unchanged. Their five-column detector matrix consists of three token features plus two verifier scores. DAPO-3B has one non-finite matrix entry and GRPO-7B has two; this is recorded rather than silently rewritten. The archive evaluator consumes the matrix directly and explicitly reports zero replacement during a new evaluation.

The public Tool Using member file has one credential-like value redacted in row 787; all smoke-test rows precede that row and are unchanged. The release manifest records the change.
