import os
import tempfile
import unittest
from datetime import date, timedelta
from unittest.mock import patch

from services import database, feature_selection


class FeatureSelectionTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.environment = patch.dict(os.environ, {
            "FEATURE_SELECTION_DIR": self.temp.name,
            "DATABASE_URL": "",
        })
        self.environment.start()
        database.reset_for_tests()

    def tearDown(self):
        database.reset_for_tests()
        self.environment.stop()
        self.temp.cleanup()

    def _record(self, count, shadow=False):
        sectors = ["電気機器", "銀行", "小売"]
        for index in range(count):
            feature_selection.record_evaluation(
                code=f"{1000 + index % 10}",
                sector=sectors[index % len(sectors)],
                model="ridge",
                market_date=str(date(2026, 1, 1) + timedelta(days=index)),
                horizon=1,
                importances=[{"feature": "noise_feature", "mse_increase": -0.01}],
                shadow_features=["noise_feature"] if shadow else [],
                shadow_noninferior=True if shadow else None,
            )

    def test_candidate_requires_repeated_cross_stock_evidence(self):
        self._record(30)
        row = feature_selection.aggregate_stats()[0]
        self.assertEqual("exclusion_candidate", row["status"])
        self.assertEqual(30, row["exclusion_count"])
        self.assertEqual(10, row["distinct_stocks"])

    def test_removed_only_after_shadow_noninferiority(self):
        self._record(30)
        self._record(20, shadow=True)
        row = feature_selection.aggregate_stats()[0]
        self.assertEqual("removed", row["status"])
        self.assertIn("noise_feature", feature_selection.get_removed_features())

    def test_duplicate_event_is_counted_once(self):
        self._record(1)
        self._record(1)
        self.assertEqual(1, feature_selection.aggregate_stats()[0]["evaluated_count"])


if __name__ == "__main__":
    unittest.main()
