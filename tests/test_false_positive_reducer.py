import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.false_positive_reducer import FalsePositiveReducer


class FalsePositiveReducerTests(unittest.TestCase):
    def test_high_model_confidence_without_file_or_network_activity_does_not_confirm(self):
        reducer = FalsePositiveReducer(required_consecutive=2, smoothing_alpha=1.0, base_threshold=0.65)
        raw = {
            "confidence": 0.95,
            "votes": {"Random Forest": 1},
            "probabilities": {"Random Forest": 0.95},
            "vote_count": 5,
            "timestamp": "2026-04-01T00:00:00",
        }
        snapshot = {
            "cpu_percent": 20.0,
            "disk_write_rate": 0.0,
            "new_process_count": 1,
            "file_modified_count": 0,
            "file_created_count": 0,
            "file_deleted_count": 0,
            "active_connections": 50,
            "bytes_sent_rate": 0.0,
            "bytes_recv_rate": 0.0,
        }

        with mock.patch("src.false_positive_reducer.get_untrusted_high_cpu_processes", return_value=[]):
            result = reducer.process(raw, snapshot)

        self.assertFalse(result["confirmed"])
        self.assertFalse(result["evidence_gate"])
        self.assertEqual(result["consecutive_count"], 0)
        self.assertLess(result["adjusted_confidence"], raw["confidence"])


if __name__ == "__main__":
    unittest.main()
