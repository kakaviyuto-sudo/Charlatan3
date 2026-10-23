from __future__ import annotations
import json
import os
from pathlib import Path

class FileStore:
    def __init__(self, base_dir: str | Path):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def load_json(self, filename: str, default=None):
        path = self.base_dir / filename
        if not path.exists():
            return default
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return default

    def write_json(self, filename: str, data) -> None:
        path = self.base_dir / filename
        temp = path.with_suffix(path.suffix + ".tmp")
        temp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(temp, path)