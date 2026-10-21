"""Create synthetic instruction data for an execution check, not membership evaluation."""
import argparse
from pathlib import Path
import pandas as pd

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--out_dir", type=Path, required=True)
args = parser.parse_args()
args.out_dir.mkdir(parents=True, exist_ok=True)
for group in ("member", "nonmember"):
    records = []
    for i in range(8):
        records.append({
            "prompt": [{"role": "user", "content": f"Write a short greeting numbered {i}. Use ALL UPPERCASE letters."}],
            "constraints": [["Language", "English: All Uppercase", "Must use all uppercase letters."]],
        })
    pd.DataFrame(records).to_parquet(args.out_dir / f"{group}.parquet", index=False)
print("Synthetic splits written. Use --skip_eval; these are not actual training memberships.")
