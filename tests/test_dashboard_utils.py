import json
import os
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import src.dashboard_utils as dashboard_utils


class DashboardUtilsTests(unittest.TestCase):
    def test_calibrate_model_probabilities_scales_display_to_context(self):
        probs = {"Random Forest": 1.0, "XGBoost": 0.8, "SVM": 0.4}
        adjusted = dashboard_utils.calibrate_model_probabilities(probs, 1.0, 0.5)

        self.assertAlmostEqual(adjusted["Random Forest"], 0.5, places=3)
        self.assertAlmostEqual(adjusted["XGBoost"], 0.4, places=3)
        self.assertAlmostEqual(adjusted["SVM"], 0.2, places=3)

    def test_load_email_config_scrubs_legacy_password(self):
        config_path = Path(__file__).resolve().parent / "_tmp_email_config.json"
        config_path.write_text(
            json.dumps(
                {
                    "sender": "from@example.com",
                    "recipient": "to@example.com",
                    "smtp_server": "smtp.gmail.com",
                    "port": 587,
                    "password": "legacy-secret",
                }
            ),
            encoding="utf-8",
        )

        original = dashboard_utils.EMAIL_CONFIG_PATH
        try:
            dashboard_utils.EMAIL_CONFIG_PATH = str(config_path)
            config = dashboard_utils.load_email_config()
            persisted = json.loads(config_path.read_text(encoding="utf-8"))
        finally:
            dashboard_utils.EMAIL_CONFIG_PATH = original
            if config_path.exists():
                os.remove(config_path)

        self.assertNotIn("password", config)
        self.assertNotIn("password", persisted)
        self.assertFalse(config_path.exists())


if __name__ == "__main__":
    unittest.main()
