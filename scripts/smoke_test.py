"""Exercise CPU evaluation with temporary synthetic features."""
from pathlib import Path
import json
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from audit import read_responses, ensure_manifest
from src.config import CONF
from src.compute_score import compute_score

assert compute_score(r"Answer: \boxed{\frac{1}{2}}", r"\frac{1}{2}") == 1.0
assert compute_score("Reasoning.\n#### 42", "42") == 1.0
assert compute_score("Reasoning.\n#### 41", "42") == 0.0
assert compute_score("No final answer.", "42") == 0.0

with tempfile.TemporaryDirectory(prefix="diba-smoke-") as directory:
    tmp = Path(directory)
    path = tmp / "responses.jsonl"
    row = {"row_id": 0, "sample_id": 0, "prompt": "p", "response": "r"}
    path.write_text(json.dumps(row) + "\n")
    assert len(read_responses(path, 1, 1)) == 1
    for contents in ["", (json.dumps(row) + "\n") * 2]:
        path.write_text(contents)
        try:
            read_responses(path, 1, 1)
        except ValueError:
            pass
        else:
            raise AssertionError("Incomplete or duplicate cache was accepted.")
    manifest_dir = tmp / "manifest"
    ensure_manifest(manifest_dir, {"seed": 1})
    try:
        ensure_manifest(manifest_dir, {"seed": 2})
    except ValueError:
        pass
    else:
        raise AssertionError("Changed cache configuration was accepted.")

    rng = np.random.default_rng(2025)
    labels = np.repeat([0, 1], 24)
    features = rng.normal(size=(48, len(CONF.base_feature_names)))
    base = rng.random(48)
    tuned = rng.random(48)
    np.savez(tmp / "metrics.npz", feats=np.column_stack([features, base, tuned]),
             labels=labels, avg_base_scores=base, avg_ft_scores=tuned,
             prompt_texts=np.array([f"synthetic-{i}" for i in range(48)]))
    subprocess.run([sys.executable, str(ROOT / "audit.py"), "--load_from_npz",
                    str(tmp / "metrics.npz"), "--out_dir", str(tmp / "evaluation")], check=True)
    report = (tmp / "evaluation/report_unified_model.txt").read_text()
    assert '"auc"' in report and "nan" not in report.lower()
    assert (tmp / "evaluation/evaluation_summary.csv").exists()

print("PASS: math answer formats, cache validation, and CPU nested-CV pipeline.")
