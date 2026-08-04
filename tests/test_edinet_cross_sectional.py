import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

APP_DIR = Path(__file__).resolve().parents[1] / "app"
sys.path.insert(0, str(APP_DIR))

from services.cross_sectional import run_cross_sectional_backtest  # noqa: E402
from services.edinet import EdinetClient, extract_financial_metrics, get_fundamental_analysis, score_fundamentals  # noqa: E402


class EdinetAndCrossSectionalTest(unittest.TestCase):
    def test_edinet_csv_metrics_are_extracted(self):
        frame = pd.DataFrame({
            "要素ID": ["NetSales", "OperatingIncome", "Assets", "Equity"],
            "コンテキストID": ["CurrentYear"] * 4,
            "値": ["1,000", "100", "2,000", "800"],
        })
        metrics = extract_financial_metrics([frame])
        self.assertEqual(metrics["revenue"], 1000.0)
        self.assertEqual(metrics["operating_income"], 100.0)
        self.assertGreater(score_fundamentals(metrics)["data_coverage"], 0)

    def test_edinet_without_api_key_degrades_gracefully(self):
        result = get_fundamental_analysis("5802", EdinetClient(api_key=""))
        self.assertFalse(result["available"])
        self.assertEqual(result["reason"], "EDINET_API_KEY_not_configured")

    def test_cross_sectional_backtest_is_lagged(self):
        index = pd.bdate_range("2020-01-01", periods=160)
        prices = pd.DataFrame({
            "1001.T": 100 * np.cumprod(np.full(len(index), 1.002)),
            "1002.T": 100 * np.cumprod(np.full(len(index), 1.001)),
            "1003.T": 100 * np.cumprod(np.full(len(index), 0.999)),
        }, index=index)
        result = run_cross_sectional_backtest(prices, top_n=1)
        self.assertEqual(result["universe_size"], 3)
        self.assertEqual(result["point_in_time_policy"], "signal_at_close_execute_next_business_day")
        self.assertIn("information_ratio", result)


if __name__ == "__main__":
    unittest.main()
