import sys
import json
import os
import tempfile
import io
import zipfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

APP_DIR = Path(__file__).resolve().parents[1] / "app"
sys.path.insert(0, str(APP_DIR))

from services.cross_sectional import run_cross_sectional_backtest  # noqa: E402
from services.edinet import EdinetClient, _read_csv_package, extract_financial_metrics, get_fundamental_analysis, score_fundamentals  # noqa: E402
from services.prime_ranking import atomic_replace_ranking, parse_prime_universe, read_latest_ranking, screen_prime_universe  # noqa: E402


class EdinetAndCrossSectionalTest(unittest.TestCase):
    def test_jpx_prime_universe_excludes_other_markets(self):
        source = pd.DataFrame({
            "コード": ["58020", "12340", "99990"],
            "銘柄名": ["A社", "B社", "ETF"],
            "市場・商品区分": ["プライム（内国株式）", "スタンダード（内国株式）", "プライム（ETF）"],
            "33業種区分": ["非鉄金属", "機械", "ETF"],
        })
        result = parse_prime_universe(source)
        self.assertEqual(result["code"].tolist(), ["5802"])

    def test_latest_ranking_is_atomically_replaced_and_read(self):
        previous = os.environ.get("PRIME_RANKING_DIR")
        try:
            with tempfile.TemporaryDirectory() as directory:
                os.environ["PRIME_RANKING_DIR"] = directory
                frame = pd.DataFrame({
                    "rank": [1, 2], "code": ["5802", "6501"],
                    "company": ["A社", "B社"], "total_score": [80.0, 70.0],
                    "analyzed_at": ["2026-08-05T06:00:00+09:00"] * 2,
                    "positive_factors": [json.dumps(["positive"])] * 2,
                    "risk_factors": [json.dumps([])] * 2,
                })
                atomic_replace_ranking(frame)
                result = read_latest_ranking(limit=1)
                self.assertTrue(result["available"])
                self.assertEqual(result["ranking"][0]["code"], "5802")
                self.assertEqual(result["ranking"][0]["positive_factors"], ["positive"])
        finally:
            if previous is None:
                os.environ.pop("PRIME_RANKING_DIR", None)
            else:
                os.environ["PRIME_RANKING_DIR"] = previous

    def test_prime_screening_uses_market_median_when_topix_is_empty(self):
        index = pd.bdate_range("2025-01-01", periods=140)
        close = pd.DataFrame({
            "5802.T": np.linspace(100, 130, len(index)),
            "6501.T": np.linspace(100, 120, len(index)),
        }, index=index)
        volume = pd.DataFrame(1000.0, index=index, columns=close.columns)
        universe = pd.DataFrame({
            "code": ["5802", "6501"], "company": ["A社", "B社"], "sector": ["非鉄金属", "電気機器"]
        })
        result = screen_prime_universe(universe, close, volume, None)
        self.assertFalse(result.empty)
        self.assertEqual(result.iloc[0]["market_benchmark_source"], "prime_universe_median_fallback")

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

    def test_edinet_utf16_tab_separated_package_is_read(self):
        text = (
            '要素ID\t項目名\tコンテキストID\t値\r\n'
            'jpcrp_cor:NetSales\t売上高\tCurrentYearDuration\t"1,000"\r\n'
            'jpcrp_cor:OperatingIncome\t営業利益\tCurrentYearDuration\t100\r\n'
        )
        package = io.BytesIO()
        with zipfile.ZipFile(package, "w") as archive:
            archive.writestr("XBRL_TO_CSV/jpcrp_test.csv", text.encode("utf-16le"))
        frames = _read_csv_package(package.getvalue())
        metrics = extract_financial_metrics(frames)
        self.assertEqual(metrics["revenue"], 1000.0)
        self.assertEqual(metrics["operating_income"], 100.0)

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
