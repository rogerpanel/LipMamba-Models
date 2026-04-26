"""RoBench-25 / RoBench-26 — hidden-state poisoning benchmark."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator


@dataclass
class RoBenchSample:
    prefix: str
    trigger: str
    family: str
    target_alpha: float
    metadata: dict


class RoBenchDataset:
    """Lightweight reader for RoBench-25 / 26 trigger benchmarks.

    Each row contains:
      * ``prefix`` — the benign context preceding the trigger
      * ``trigger`` — the (potentially adversarial) suffix
      * ``family`` — one of nine attack families (e.g. ``zeroing``,
        ``flip``, ``replay``, ``echo`` ...)
      * ``target_alpha`` — the α threshold the attacker tries to achieve
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f"RoBench file {self.path} not found.")

    def __iter__(self) -> Iterator[RoBenchSample]:
        with open(self.path, "r", encoding="utf-8") as fh:
            for line in fh:
                obj = json.loads(line)
                yield RoBenchSample(
                    prefix=obj["prefix"],
                    trigger=obj["trigger"],
                    family=obj.get("family", "unknown"),
                    target_alpha=float(obj.get("target_alpha", 0.05)),
                    metadata={k: v for k, v in obj.items() if k not in {"prefix", "trigger"}},
                )

    def families(self) -> set[str]:
        return {s.family for s in self}
