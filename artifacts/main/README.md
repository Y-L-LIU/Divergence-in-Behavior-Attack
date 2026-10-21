# Archived main MATH experiments

This release preserves the four original MATH runs (GRPO/DAPO × 3B/7B) from the experiment archive. It includes the saved metrics here and the generated target responses and sample features in [RLMIA/DIBA-MATH-RESULTS](https://huggingface.co/datasets/RLMIA/DIBA-MATH-RESULTS).

Each run contains 1,269 member and 1,269 nonmember prompts, with 8 target responses per prompt. The archive directories record temperature **0.5**. These are archived runs, separate from the default temperature 0.7 of the portable runner.

## Download generated responses

```bash
hf download RLMIA/DIBA-MATH-RESULTS --repo-type dataset \
  --revision b835b05c6dafed8b2011f6f3a7bda4fbc5cd6e18 --local-dir runs/released-main
gzip -dk runs/released-main/grpo-3b/responses_ft_member.jsonl.gz
```

The dataset contains only the four main runs. Model weights are in their model repositories. Tuning runs, ablations, intermediate checkpoints, and training logs are excluded.

## Contents

Each run has the original `metrics_prompt.npz`, `evaluation_summary.csv`, and the unified/specialist reports. The Hugging Face dataset additionally contains compressed target-response JSONL files for members/nonmembers and `computed_features.jsonl.gz`. [manifest.json](manifest.json) records hashes, row counts, feature shape, and non-finite counts.

The original archive contains **no reference-response JSONL files**. It does contain the saved reference scores in both the sample feature cache and prompt feature cache. Recomputing verifier scores from both models requires generating the missing reference responses with the corresponding original settings.

## Feature layout and evaluation

The archived `feats` array has five columns, in this order:

1. `neg_log_likelihood`
2. `zlib_entropy_ratio`
3. `reverse_kl_div`
4. `avg_base_score`
5. `avg_ft_score`

The last two columns are verifier scores. This is a different layout from the portable runner's seven-column output. Use the evaluator that consumes the saved numeric array directly:

```bash
python scripts/evaluate_cached.py \
  --input artifacts/main/grpo-3b/metrics_prompt.npz \
  --out_dir runs/released-grpo-3b-reeval
```

This fits the current unified predictor on the saved feature matrix; original reports remain unchanged. Add `--limit_per_class 8` for an execution check only.

Original values are preserved: DAPO-3B contains one non-finite detector-feature entry and GRPO-7B contains two. The raw sample-feature caches also contain non-finite values in unused features. The evaluator reports how many detector-feature entries it replaces with zero, following the current evaluation pipeline; it does not modify the archive. The historical reports and a new evaluation should be identified separately.
