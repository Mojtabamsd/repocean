from __future__ import annotations
from pathlib import Path
from typing import Dict
import re
import yaml
from yaml.constructor import ConstructorError


def read_yaml_relaxed(yaml_path: str | Path) -> Dict:
    """Safe-load YAML; if python/object tags exist, strip and retry."""
    p = Path(yaml_path)
    text = p.read_text(encoding="utf-8")
    try:
        return yaml.safe_load(text) or {}
    except ConstructorError:
        pass
    cleaned = re.sub(r"!!python/object:[^\s]+", "", text)
    cleaned = re.sub(r"tag:yaml\.org,2002:python/object:[^\s]+", "", cleaned)
    cleaned = re.sub(r"!!python/[^\s]+", "", cleaned)
    return yaml.safe_load(cleaned) or {}


def load_run_config(yaml_path: str | Path) -> Dict:
    raw = read_yaml_relaxed(yaml_path)
    input_path = raw.get("input_path") or raw.get("input-dir") or raw.get("data_path")
    return {"input_path": str(Path(input_path)) if input_path else None}
