"""Verifier and data-schema checks; no model weights required."""
from pathlib import Path
import tempfile
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from audit import load_data
from src.utils import compute_score_safe, load_prompt_data_from_npz

truth = '<tool_call>\n{"name":"lookup","parameters":{"city":"Seoul"}}\n</tool_call>'
wrong = '<tool_call>\n{"name":"lookup","parameters":{"city":"Paris"}}\n</tool_call>'
assert compute_score_safe(truth, truth, "toolrl") == 1.0
assert 0 <= compute_score_safe(wrong, truth, "toolrl") < 1.0
assert compute_score_safe("<tool_call>invalid</tool_call>", truth, "toolrl") == 0.0
assert compute_score_safe("No call.", truth, "toolrl") == 0.0

constraints = [["Language", "English: All Uppercase", "Must use all uppercase letters."]]
assert compute_score_safe("HELLO WORLD", constraints, "if") == 1.0
assert compute_score_safe("hello world", constraints, "if") == 0.0
try:
    compute_score_safe("HELLO", [], "if")
except ValueError:
    pass
else:
    raise AssertionError("Empty constraints accepted.")

with tempfile.TemporaryDirectory() as d:
    path = Path(d) / "instruction.parquet"
    pd.DataFrame([{"prompt":[{"role":"user","content":"Say hello."}],
                   "constraints":constraints}]).to_parquet(path,index=False)
    assert len(load_data(path, None, "instruction")) == 1
    try:
        load_data(path, None, "math")
    except ValueError:
        pass
    else:
        raise AssertionError("Instruction data was accepted as MATH.")
    matrix = np.ones((8,5))
    cache = Path(d) / "legacy.npz"
    np.savez(cache, feats=matrix, labels=np.arange(8)%2,
             avg_base_scores=matrix[:,-2], avg_ft_scores=matrix[:,-1])
    try:
        load_prompt_data_from_npz(cache)
    except ValueError as e:
        assert "evaluate_cached.py" in str(e)
    else:
        raise AssertionError("Legacy score columns misinterpreted as token features.")

print("PASS: Tool Using, Instruction Following, schemas, legacy cache guard.")
