"""Small retry helpers for long-running watchers on the research-drive mount."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path


RETRY_DELAYS_SECONDS = (0.0, 0.25, 0.5, 1.0, 2.0, 2.0, 2.0, 2.0)


def read_json_retry(path: Path) -> dict:
    last_error = None
    for delay in RETRY_DELAYS_SECONDS:
        if delay:
            time.sleep(delay)
        try:
            with path.open("r", encoding="utf-8") as stream:
                return json.load(stream)
        except (OSError, json.JSONDecodeError) as error:
            last_error = error
    raise last_error


def write_json_atomic_retry(path: Path, payload: dict) -> None:
    partial = path.with_name(f"{path.name}.partial")
    last_error = None
    for delay in RETRY_DELAYS_SECONDS:
        if delay:
            time.sleep(delay)
        try:
            with partial.open("w", encoding="utf-8") as stream:
                json.dump(payload, stream, indent=2)
                stream.write("\n")
            os.replace(partial, path)
            return
        except OSError as error:
            last_error = error
    raise last_error


def count_lines_retry(path: Path) -> int:
    if not path.is_file():
        return 0
    last_error = None
    for delay in RETRY_DELAYS_SECONDS:
        if delay:
            time.sleep(delay)
        try:
            with path.open("rb") as stream:
                return sum(1 for _ in stream)
        except OSError as error:
            last_error = error
    raise last_error
