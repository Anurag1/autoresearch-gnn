"""CPU-only integration tests for research_tracker."""
import json
import tempfile
import unittest
from pathlib import Path

from research_tracker import build_parser, parse_metrics, record_run


class MetricParsingTests(unittest.TestCase):
    def test_parses_autoresearch_summary_and_ignores_noise(self):
        metrics = parse_metrics(
            "noise\nval_bpb:          0.997900\n"
            "training_seconds: 300.1\npeak_vram_mb: 45056.0\n"
            "num_steps: 953\nnot_a_metric: 8\n"
        )
        self.assertEqual(metrics["val_bpb"], 0.9979)
        self.assertEqual(metrics["training_seconds"], 300.1)
        self.assertEqual(metrics["peak_vram_mb"], 45056.0)
        self.assertNotIn("not_a_metric", metrics)


class RecordRunTests(unittest.TestCase):
    def _args(self, directory, log, config, **extra):
        values = [
            "record", "--run-log", str(log), "--config-json", str(config),
            "--commit", "abcdef123456", "--description", "test experiment",
            "--results-tsv", str(directory / "results.tsv"),
            "--ledger", str(directory / "research" / "experiments.jsonl"),
            "--evidence", str(directory / "research" / "evidence.jsonl"),
            "--train-script", str(directory / "train.py"),
        ]
        for key, value in extra.items():
            values.extend(["--" + key.replace("_", "-"), str(value)])
        return build_parser().parse_args(values)

    def test_records_baseline_and_writes_tsv_and_ledger(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            log = directory / "run.log"
            log.write_text("val_bpb: 1.25\npeak_vram_mb: 2048\n", encoding="utf-8")
            config = directory / "config.json"
            config.write_text(json.dumps({"lr": 0.02}), encoding="utf-8")
            args = self._args(directory, log, config)
            self.assertEqual(record_run(args), 0)
            rows = (directory / "results.tsv").read_text(encoding="utf-8").splitlines()
            self.assertEqual(rows[0], "commit\tval_bpb\tmemory_gb\tstatus\tdescription")
            self.assertIn("abcdef1\t1.250000\t2.0\tkeep\ttest experiment", rows[1])
            run = json.loads((directory / "research" / "experiments.jsonl").read_text().splitlines()[0])
            self.assertEqual(run["metrics"]["val_bpb"], 1.25)
            self.assertEqual(run["status"], "keep")

    def test_lower_bpb_keeps_and_records_supporting_evidence(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            log = directory / "run.log"
            log.write_text("val_bpb: 0.9\npeak_vram_mb: 1024\n", encoding="utf-8")
            config = directory / "config.json"
            config.write_text('{"depth": 8}', encoding="utf-8")
            args = self._args(directory, log, config, baseline_val_bpb=1.0)
            self.assertEqual(record_run(args), 0)
            evidence = (directory / "research" / "evidence.jsonl").read_text().splitlines()
            self.assertEqual(json.loads(evidence[0])["stance"], "supports")

    def test_missing_metric_records_crash_and_returns_nonzero(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            log = directory / "run.log"
            log.write_text("RuntimeError: OOM\n", encoding="utf-8")
            config = directory / "config.json"
            config.write_text("{}", encoding="utf-8")
            args = self._args(directory, log, config)
            self.assertEqual(record_run(args), 2)
            run = json.loads((directory / "research" / "experiments.jsonl").read_text().splitlines()[0])
            self.assertEqual(run["status"], "crash")


if __name__ == "__main__":
    unittest.main()
