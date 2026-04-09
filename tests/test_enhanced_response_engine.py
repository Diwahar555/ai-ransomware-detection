import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import src.enhanced_response_engine as response_engine


class EnhancedResponseEngineTests(unittest.TestCase):
    def test_protect_files_records_failed_icacls_calls(self):
        completed = mock.Mock(returncode=5, stderr="denied", stdout="")
        with mock.patch.object(response_engine, "SENSITIVE_FOLDERS", ["C:\\fake"]), \
             mock.patch("src.enhanced_response_engine.os.path.exists", return_value=True), \
             mock.patch("src.enhanced_response_engine.subprocess.run", return_value=completed):
            result = response_engine.protect_files(True)

        self.assertFalse(result["success"])
        self.assertEqual(len(result["failed"]), 1)
        self.assertEqual(result["failed"][0]["folder"], "C:\\fake")

    def test_respond_to_threat_skips_low_risk_process(self):
        candidate = {
            "pid": 1234,
            "name": "odd.exe",
            "cpu": 12.0,
            "memory": 4.0,
            "risk_score": response_engine.AUTO_QUARANTINE_SCORE - 1,
            "reasons": ["Running from a user-writable path"],
        }
        with mock.patch("src.enhanced_response_engine.find_suspicious_process", return_value=candidate), \
             mock.patch("src.enhanced_response_engine.send_notification", return_value=False), \
             mock.patch("src.enhanced_response_engine.log_incident"):
            result = response_engine.respond_to_threat({"confidence": 0.9, "vote_count": 5})

        self.assertEqual(result["quarantine"]["action"], "skipped")
        self.assertFalse(result["quarantine"]["success"])
        self.assertIn("threshold", result["quarantine"]["message"])


if __name__ == "__main__":
    unittest.main()
