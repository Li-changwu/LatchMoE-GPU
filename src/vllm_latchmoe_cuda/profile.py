from __future__ import annotations

import json
import threading
import time
from collections import Counter
from pathlib import Path
from typing import Any


class RuntimeCounters:
    def __init__(self) -> None:
        self._counts: Counter[str] = Counter()
        self._lock = threading.Lock()

    def increment(self, name: str, value: int = 1) -> None:
        with self._lock:
            self._counts[name] += int(value)

    def snapshot(self) -> dict[str, int]:
        with self._lock:
            return dict(self._counts)


class JsonlEventWriter:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self.path.open("a", encoding="utf-8")
        self._lock = threading.Lock()
        self._closed = False

    def write(self, event: str, **fields: Any) -> None:
        payload = {"event": event, "timestamp_ns": time.time_ns(), **fields}
        line = json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n"
        with self._lock:
            if self._closed:
                raise RuntimeError("JSONL writer is closed")
            self._handle.write(line)
            self._handle.flush()

    def close(self) -> None:
        with self._lock:
            if not self._closed:
                self._handle.flush()
                self._handle.close()
                self._closed = True

    def __enter__(self) -> JsonlEventWriter:
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()
