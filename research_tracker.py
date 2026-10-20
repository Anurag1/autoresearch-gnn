"""Record autoresearch runs without modifying the training/evaluation harness.

Example:
  python research_tracker.py record --run-log run.log --config-json experiment_config.json \
      --commit "$(git rev-parse --short HEAD)" --description "baseline"

This script records results after a run; it does not launch training or claim
that graph memory improves model quality.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from discovery_memory import DiscoveryMemory, Evidence

METRIC_NAMES = (
    "val_bpb", "training_seconds", "total_seconds", "peak_vram_mb",
    "mfu_percent", "total_tokens_M", "num_steps", "num_params_M", "depth",
)
RESULTS_HEADER = "commit\tval_bpb\tmemory_gb\tstatus\tdescription\n"


def parse_metrics(log_text: str) -> dict[str, float]:
    """Extract numeric metrics printed as 'metric_name: number'."""
    metrics: dict[str, float] = {}
    wanted = set(METRIC_NAMES)
    pattern = re.compile(r"^\s*([A-Za-z][A-Za-z0-9_]*):\s*([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)\s*$")
    for line in log_text.splitlines():
        match = pattern.match(line)
        if not match or match.group(1) not in wanted:
            continue
        value = float(match.group(2))
        if math.isfinite(value):
            metrics[match.group(1)] = value
    return metrics


def _append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, sort_keys=True, ensure_ascii=False) + "\n")


def _sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def record_run(args: argparse.Namespace) -> int:
    run_log = Path(args.run_log)
    config_path = Path(args.config_json)
    if not run_log.is_file():
        raise FileNotFoundError(f"run log not found: {run_log}")
    if not config_path.is_file():
        raise FileNotFoundError(f"config JSON not found: {config_path}")

    config = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("config JSON must contain an object")

    metrics = parse_metrics(run_log.read_text(encoding="utf-8", errors="replace"))
    val_bpb = metrics.get("val_bpb")
    if val_bpb is None:
        status = "crash"
    elif args.baseline_val_bpb is None:
        status = "keep"
    else:
        status = "keep" if val_bpb < args.baseline_val_bpb else "discard"

    canonical_config = json.dumps(config, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    config_hash = hashlib.sha256(canonical_config.encode("utf-8")).hexdigest()
    run_record: dict[str, Any] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "commit": args.commit,
        "config_sha256": config_hash,
        "config": config,
        "description": args.description,
        "status": status,
        "metrics": metrics,
        "baseline_val_bpb": args.baseline_val_bpb,
        "train_py_sha256": _sha256(Path(args.train_script)),
    }
    _append_jsonl(Path(args.ledger), run_record)

    results_path = Path(args.results_tsv)
    results_path.parent.mkdir(parents=True, exist_ok=True)
    if not results_path.exists() or results_path.stat().st_size == 0:
        results_path.write_text(RESULTS_HEADER, encoding="utf-8")
    memory_gb = metrics.get("peak_vram_mb", 0.0) / 1024.0
    logged_bpb = val_bpb if val_bpb is not None else 0.0
    description = args.description.replace("\t", " ").replace("\n", " ")
    with results_path.open("a", encoding="utf-8") as stream:
        stream.write(f"{args.commit[:7]}\t{logged_bpb:.6f}\t{memory_gb:.1f}\t{status}\t{description}\n")

    if args.baseline_val_bpb is not None and val_bpb is not None:
        evidence_path = Path(args.evidence)
        existing = DiscoveryMemory.load_jsonl(evidence_path) if evidence_path.exists() else DiscoveryMemory()
        existing.add(Evidence(
            subject=f"config:{config_hash}",
            predicate="beats baseline val_bpb",
            value=f"val_bpb={val_bpb:.6f}; baseline={args.baseline_val_bpb:.6f}",
            stance="supports" if val_bpb < args.baseline_val_bpb else "refutes",
            source=args.commit,
            confidence=1.0,
            experiment_id=config_hash[:16],
        ))
        existing.save_jsonl(evidence_path)

    print(json.dumps({
        "commit": args.commit,
        "status": status,
        "val_bpb": val_bpb,
        "baseline_val_bpb": args.baseline_val_bpb,
        "config_sha256": config_hash,
        "ledger": str(args.ledger),
        "results_tsv": str(results_path),
    }, indent=2, sort_keys=True))
    return 0 if status != "crash" else 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    record = subparsers.add_parser("record", help="record a completed or crashed training run")
    record.add_argument("--run-log", required=True, help="path to captured train.py output")
    record.add_argument("--config-json", required=True, help="JSON object describing this experiment's configuration")
    record.add_argument("--commit", required=True, help="git commit SHA for the experiment")
    record.add_argument("--description", required=True, help="short description of the experiment")
    record.add_argument("--baseline-val-bpb", type=float, help="baseline val_bpb; lower is better")
    record.add_argument("--train-script", default="train.py", help="training script to fingerprint (default: train.py)")
    record.add_argument("--results-tsv", default="results.tsv", help="TSV output (default: results.tsv; keep untracked)")
    record.add_argument("--ledger", default="research/experiments.jsonl", help="append-only run ledger")
    record.add_argument("--evidence", default="research/evidence.jsonl", help="contradiction-aware evidence store")
    record.set_defaults(func=record_run)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"research_tracker: error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
