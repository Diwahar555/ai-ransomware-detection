import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.generate_500k_dataset import (
    FEATURE_NAMES,
    build_generation_profile,
    gen_slow_encryption,
    introduce_class_overlap,
    synthesize_normal,
)


class GenerateDatasetTests(unittest.TestCase):
    def setUp(self):
        rows = []
        for index in range(60):
            rows.append(
                {
                    "cpu_percent": 10 + (index % 5),
                    "memory_percent": 35 + (index % 7),
                    "process_count": 90 + (index % 9),
                    "high_cpu_process_count": 1 + (index % 3),
                    "active_connections": 15 + (index % 6),
                    "established_connections": 8 + (index % 4),
                    "unique_remote_ports": 4 + (index % 3),
                    "bytes_sent_rate": 1000 + index * 25,
                    "bytes_recv_rate": 2000 + index * 30,
                    "file_modified_count": index % 4,
                    "file_created_count": index % 3,
                    "file_deleted_count": index % 2,
                    "disk_write_rate": 50000 + index * 1000,
                    "new_process_count": index % 3,
                }
            )
        self.normal_df = pd.DataFrame(rows, columns=FEATURE_NAMES)

    def test_overlap_generation_preserves_basic_invariants(self):
        profile = build_generation_profile(self.normal_df)
        normal = synthesize_normal(profile, 40, chunk_size=20)
        attack = gen_slow_encryption(profile, 30)
        normal, attack, _, _ = introduce_class_overlap(normal, attack, profile)

        for frame in [normal, attack]:
            self.assertTrue((frame["active_connections"] >= frame["established_connections"]).all())
            self.assertTrue((frame["process_count"] >= frame["high_cpu_process_count"]).all())
            self.assertTrue((frame["process_count"] >= frame["new_process_count"]).all())
            numeric = frame[FEATURE_NAMES].to_numpy(dtype=np.float64)
            self.assertTrue((numeric >= 0).all())


if __name__ == "__main__":
    unittest.main()
