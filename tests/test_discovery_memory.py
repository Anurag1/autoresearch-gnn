"""Unit tests for graph-memory discovery primitives; no GPU or model downloads."""
import tempfile
import unittest
from pathlib import Path

from discovery_memory import DiscoveryMemory, Evidence, Experiment, ExperimentLedger


class DiscoveryMemoryTests(unittest.TestCase):
    def test_detects_opposing_evidence_for_same_claim(self):
        memory = DiscoveryMemory([
            Evidence("attention", "improves validation BPB", "yes", "supports", "run-1", 0.8),
            Evidence(" Attention ", "IMPROVES validation BPB", "no", "refutes", "run-2", 0.7),
        ])
        contradictions = memory.contradictions()
        self.assertEqual(len(contradictions), 1)
        self.assertEqual(contradictions[0].claim_key,
                         ("attention", "improves validation bpb"))

    def test_does_not_flag_different_claims_or_neutral_evidence(self):
        memory = DiscoveryMemory([
            Evidence("model A", "improves BPB", "yes", "supports", "run-1"),
            Evidence("model B", "improves BPB", "no", "refutes", "run-2"),
            Evidence("model A", "improves BPB", "unknown", "neutral", "run-3"),
        ])
        self.assertEqual(memory.contradictions(), [])

    def test_round_trip_jsonl(self):
        memory = DiscoveryMemory([
            Evidence("optimizer", "helps", "yes", "supports", "run-9", 0.9, "exp-9")
        ])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "nested" / "evidence.jsonl"
            memory.save_jsonl(path)
            loaded = DiscoveryMemory.load_jsonl(path)
        self.assertEqual(loaded.evidence, memory.evidence)

    def test_rejects_invalid_confidence(self):
        with self.assertRaises(ValueError):
            Evidence("x", "y", "z", "supports", "source", 1.1)


class ExperimentLedgerTests(unittest.TestCase):
    def test_config_order_does_not_change_fingerprint(self):
        a = Experiment({"lr": 0.01, "depth": 4}, "val_bpb")
        b = Experiment({"depth": 4, "lr": 0.01}, "val_bpb")
        self.assertEqual(a.fingerprint, b.fingerprint)

    def test_deduplicates_config_and_selects_best_metric(self):
        ledger = ExperimentLedger()
        self.assertTrue(ledger.record(Experiment({"lr": 0.01}, "val_bpb")))
        self.assertFalse(ledger.record(Experiment({"lr": 0.01}, "val_bpb", notes="duplicate")))
        ledger.record(Experiment({"lr": 0.02}, "val_bpb", 1.2, "completed"))
        ledger.record(Experiment({"lr": 0.03}, "val_bpb", 1.1, "completed"))
        self.assertEqual(ledger.best().metric_value, 1.1)
        self.assertEqual(len(ledger.experiments), 3)

    def test_completed_requires_metric(self):
        with self.assertRaises(ValueError):
            ExperimentLedger([Experiment({"lr": 0.1}, "val_bpb", None, "completed")])


if __name__ == "__main__":
    unittest.main()
