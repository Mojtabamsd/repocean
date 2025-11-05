from __future__ import annotations
from pathlib import Path
import pandas as pd

REQUIRED = {
    "features": "features_*.h5",
    "preds": "predictions_with_top3_scores.csv",
    "model_cfg": "model_config.yaml",
    "run_cfg": "config.yaml",
}


def build_run_index(parent_dir: str | Path) -> pd.DataFrame:
    parent = Path(parent_dir)
    rows = []
    for sub in parent.iterdir():
        if not sub.is_dir():
            continue
        hit = {"run_id": sub.name, "run_dir": str(sub)}
        ok = True
        for k, pat in REQUIRED.items():
            matches = list(sub.glob(pat))
            if not matches:
                ok = False
                break
            hit[k] = str(matches[0])
        if ok:
            rows.append(hit)
    df = pd.DataFrame(rows).sort_values("run_id").reset_index(drop=True)
    return df
