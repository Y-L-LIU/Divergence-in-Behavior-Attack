# Tool Using audit data

These are release copies of the split files used by the server ToolRL XLAM audit:

- `member.parquet`: 1,498 prompts from the ToolRL rlla_4k training data.
- `nonmember_xlam.parquet`: 1,500 prompts in the server XLAM nonmember split.

Each row contains chat messages in `prompt` and an expected response/tool-call string in `reward_model.ground_truth`. Tool names, argument schemas, and all conversation turns are retained.

The scorer checks tool names and arguments without executing the calls. Its value is the correctness component used for auditing, not the weighted combination of correctness, format, and length used by the training plugin.

[manifest.json](manifest.json) records exact hashes and row counts. The original nonmember construction recipe is not included; these files preserve the actual saved split rather than constructing a new one. Use the corresponding Tool Using target, and do not infer membership for unrelated models from these filenames.

The public member copy replaces one credential-like GitHub access-token default in row 787 with `REDACTED_GITHUB_TOKEN`. All other records and the nonmember file are preserved. The manifest records the source checksum and redaction; the server source is unchanged. This sanitized prompt differs from the original training input.
