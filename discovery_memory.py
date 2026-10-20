"""Lightweight graph memory for autoresearch experiment selection.

This module is deliberately independent of train.py and prepare.py. It records
claims and experiment outcomes without changing the fixed training/evaluation
harness. Standard-library only.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Iterable, Literal

Stance = Literal["supports", "refutes", "neutral"]


def _norm(value: str) -> str:
    return " ".join(value.casefold().split())


@dataclass(frozen=True)
class Evidence:
    """One sourced observation about a normalized claim."""
    subject: str
    predicate: str
    value: str
    stance: Stance
    source: str
    confidence: float = 0.5
    experiment_id: str = ""

    def __post_init__(self) -> None:
        if not self.subject.strip() or not self.predicate.strip():
            raise ValueError("subject and predicate must be non-empty")
        if not self.source.strip():
            raise ValueError("source must be non-empty")
        if self.stance not in ("supports", "refutes", "neutral"):
            raise ValueError("stance must be supports, refutes, or neutral")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be between 0 and 1")

    @property
    def claim_key(self) -> tuple[str, str]:
        return (_norm(self.subject), _norm(self.predicate))


@dataclass(frozen=True)
class Contradiction:
    claim_key: tuple[str, str]
    left: Evidence
    right: Evidence
    reason: str


class DiscoveryMemory:
    """Append-only evidence memory with deterministic contradiction checks."""

    def __init__(self, evidence: Iterable[Evidence] = ()) -> None:
        self._evidence: list[Evidence] = list(evidence)

    @property
    def evidence(self) -> tuple[Evidence, ...]:
        return tuple(self._evidence)

    def add(self, item: Evidence) -> None:
        self._evidence.append(item)

    def contradictions(self) -> list[Contradiction]:
        found: list[Contradiction] = []
        by_claim: dict[tuple[str, str], list[Evidence]] = {}
        for item in self._evidence:
            by_claim.setdefault(item.claim_key, []).append(item)
        for key, items in by_claim.items():
            for i, left in enumerate(items):
                for right in items[i + 1:]:
                    if {left.stance, right.stance} == {"supports", "refutes"}:
                        found.append(Contradiction(
                            claim_key=key,
                            left=left,
                            right=right,
                            reason="opposing stances on the same normalized claim",
                        ))
        return found

    def save_jsonl(self, path: str | Path) -> None:
        """Persist evidence as UTF-8 JSONL; parent directories are created."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w", encoding="utf-8") as stream:
            for item in self._evidence:
                stream.write(json.dumps(asdict(item), ensure_ascii=False, sort_keys=True))
                stream.write("\n")

    @classmethod
    def load_jsonl(cls, path: str | Path) -> "DiscoveryMemory":
        items: list[Evidence] = []
        with Path(path).open("r", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.strip():
                    continue
                try:
                    items.append(Evidence(**json.loads(line)))
                except (TypeError, ValueError, json.JSONDecodeError) as exc:
                    raise ValueError(f"invalid evidence at line {line_number}: {exc}") from exc
        return cls(items)


@dataclass(frozen=True)
class Experiment:
    """Canonical experiment specification and measured outcome."""
    config: dict
    metric_name: str
    metric_value: float | None = None
    status: Literal["proposed", "completed", "failed"] = "proposed"
    notes: str = ""

    @property
    def fingerprint(self) -> str:
        canonical = json.dumps(self.config, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


class ExperimentLedger:
    """Deduplicate experiment configurations and preserve recorded outcomes."""

    def __init__(self, experiments: Iterable[Experiment] = ()) -> None:
        self._experiments: dict[str, Experiment] = {}
        for experiment in experiments:
            self.record(experiment)

    @property
    def experiments(self) -> tuple[Experiment, ...]:
        return tuple(self._experiments.values())

    def record(self, experiment: Experiment) -> bool:
        """Store a new config; return False rather than duplicate an existing one."""
        key = experiment.fingerprint
        if key in self._experiments:
            return False
        if experiment.status == "completed" and experiment.metric_value is None:
            raise ValueError("completed experiments require a metric_value")
        self._experiments[key] = experiment
        return True

    def contains_config(self, config: dict) -> bool:
        return Experiment(config=config, metric_name="").fingerprint in self._experiments

    def best(self, *, lower_is_better: bool = True) -> Experiment | None:
        completed = [e for e in self._experiments.values()
                     if e.status == "completed" and e.metric_value is not None]
        if not completed:
            return None
        return min(completed, key=lambda e: e.metric_value) if lower_is_better else max(
            completed, key=lambda e: e.metric_value
        )

    def save_jsonl(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w", encoding="utf-8") as stream:
            for experiment in self._experiments.values():
                stream.write(json.dumps(asdict(experiment), ensure_ascii=False, sort_keys=True))
                stream.write("\n")
