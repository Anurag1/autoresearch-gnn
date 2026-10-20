# Graph-Memory Discovery Layer for autoresearch

This branch adds a small, standard-library-only prototype for recording evidence
and experiment configurations around the existing autoresearch loop. It does
**not** change `train.py`, `prepare.py`, the fixed validation metric, or GPU
training behavior.

## Why this exists

Autonomous research loops can repeat equivalent configurations and lose the
reasoning behind conflicting outcomes. This prototype provides two primitives:

- **Evidence memory:** normalized claim keys and explicit `supports` / `refutes`
  stances, with deterministic contradiction pairs.
- **Experiment ledger:** stable fingerprints for JSON-compatible configurations,
  duplicate detection, and selection of the best completed metric.

These are bookkeeping and decision-support primitives, not a claim that a
language model's reasoning has improved. Contradictions are flagged only when
records use the same normalized subject/predicate and opposing stances. Human
review is still required to decide whether evidence is genuinely comparable.

## Quick check

Requires Python 3.10+ and no third-party packages:

```bash
python -m unittest discover -s tests -v
```

## Minimal usage

```python
from discovery_memory import DiscoveryMemory, Evidence, Experiment, ExperimentLedger

memory = DiscoveryMemory()
memory.add(Evidence(
    subject="candidate architecture",
    predicate="improves val_bpb",
    value="yes",
    stance="supports",
    source="run-001",
    confidence=0.7,
))
memory.add(Evidence(
    subject="candidate architecture",
    predicate="improves val_bpb",
    value="no",
    stance="refutes",
    source="run-002",
    confidence=0.6,
))
print(memory.contradictions())
memory.save_jsonl("research/evidence.jsonl")

ledger = ExperimentLedger()
ledger.record(Experiment({"depth": 8, "lr": 0.02}, "val_bpb"))
print(ledger.contains_config({"lr": 0.02, "depth": 8}))
```

## Integration boundary

1. Establish the existing training baseline first.
2. Record each run's immutable config, code revision, metric, runtime, and peak
   memory in a separate ledger.
3. Compare the candidate against the baseline using a fixed evaluation set and
   equal compute budgets.
4. Promote an experiment only after checking the metric and regression risks.
5. Do not let evidence memory modify the evaluation harness or silently change
   experiment acceptance rules.

The initial unit tests validate the bookkeeping behavior only. No GPU training
or performance claim is implied by passing these tests.


## Run-loop integration

`research_tracker.py` records real `train.py` output after a run. It extracts
the existing printed metrics, hashes the supplied JSON configuration, records
the commit and training-script hash, appends the standard `results.tsv` row,
and stores machine-readable metadata in `research/experiments.jsonl`. When a
baseline `val_bpb` is provided, it also writes a `supports` or `refutes`
evidence record to `research/evidence.jsonl`. Repeated conflicting outcomes
for the same configuration become inspectable as contradictory evidence.

Example baseline workflow:

```bash
python research_tracker.py record --run-log run.log \
  --config-json experiment_config.json \
  --commit "$(git rev-parse --short HEAD)" --description "baseline"
```

For a candidate, add `--baseline-val-bpb <measured-baseline>`. The script does
not launch training and does not change the evaluation harness. Keep generated
logs and local ledgers out of commits unless intentionally preparing a
sanitized results release.

## Validation boundary

CPU tests validate metric parsing and artifact writing with synthetic logs.
They do not substitute for a GPU experiment. Scientific conclusions require
repeated baseline/candidate runs, matched training budgets and data, recording
hardware/software details, and reporting variance as well as the best result.
