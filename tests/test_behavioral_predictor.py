import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.behavioral_predictor import FEATURE_NAMES, predict_behavioral


class IdentityScaler:
    def transform(self, values):
        return values


class ProbModel:
    def __init__(self, prob):
        self.prob = prob

    def predict_proba(self, _):
        return np.array([[1 - self.prob, self.prob]], dtype=np.float32)


class NNModel:
    def __init__(self, prob):
        self.prob = prob

    def predict(self, _, verbose=0):
        return np.array([[self.prob]], dtype=np.float32)


class MetaModel:
    def __init__(self, prob):
        self.prob = prob

    def predict_proba(self, _):
        return np.array([[1 - self.prob, self.prob]], dtype=np.float32)


class BehavioralPredictorTests(unittest.TestCase):
    def setUp(self):
        self.snapshot = {feature: 1.0 for feature in FEATURE_NAMES}
        self.models = {
            "Random Forest": ProbModel(0.10),
            "XGBoost": ProbModel(0.20),
            "SVM": ProbModel(0.30),
            "DNN": NNModel(0.40),
            "LSTM": NNModel(0.50),
        }
        self.scaler = IdentityScaler()

    def test_predict_behavioral_uses_meta_ensemble_when_available(self):
        result = predict_behavioral(
            self.snapshot,
            self.models,
            self.scaler,
            threshold=0.5,
            ensemble_model=MetaModel(0.91),
        )
        self.assertEqual(result["ensemble_method"], "calibrated_meta")
        self.assertTrue(result["is_threat"])
        self.assertAlmostEqual(result["confidence"], 0.91, places=2)

    def test_predict_behavioral_falls_back_to_weighted_ensemble(self):
        result = predict_behavioral(
            self.snapshot,
            self.models,
            self.scaler,
            threshold=0.5,
            ensemble_model=None,
        )
        self.assertEqual(result["ensemble_method"], "weighted_fallback")
        self.assertIn("Random Forest", result["probabilities"])
        self.assertGreaterEqual(result["vote_count"], 0)


if __name__ == "__main__":
    unittest.main()
