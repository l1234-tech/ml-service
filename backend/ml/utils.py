import json
from pathlib import Path
from typing import Any

import joblib


def save_object(obj: Any, path: Path) -> None:
    joblib.dump(obj, path)


def load_object(path: Path) -> Any:
    return joblib.load(path)


def save_json(data: Any, path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
