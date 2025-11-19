"""
Utility helpers for organizing experiment outputs.

Each script can call ``create_results_dir`` to get a fresh directory inside
``./out`` where it can dump plots, CSVs, JSON, etc., without cluttering the
repository root.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Union
from uuid import uuid4

PathLike = Union[str, Path]

def create_results_dir(tag: str, root: PathLike = "out") -> Path:
    """
    Create and return a unique results directory inside ``root``.

    The directory name encodes the provided ``tag``, a timestamp, and a short
    random suffix so repeated runs within the same second never collide.
    """
    base = Path(root)
    base.mkdir(parents=True, exist_ok=True)

    safe_tag = "".join(c if (c.isalnum() or c in ("-", "_")) else "_" for c in tag.strip())
    safe_tag = safe_tag or "run"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = uuid4().hex[:6]
    run_dir = base / f"{safe_tag}_{timestamp}_{suffix}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir

__all__ = ["create_results_dir"]
