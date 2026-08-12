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
from services.prime_ranking import _bulk_symbol_frame, atomic_replace_ranking, now_jst, parse_prime_universe, read_latest_ranking, read_status, rerank_enriched_candidates, screen_prime_universe, sector_etf_symbol  # noqa: E402
from services import hybrid_model  # noqa: E402


class EdinetAndCrossSectionalTest(unittest.TestCase):
    def test_global_model_is_saved_and_predicts_with_sector_correction(self):
        previous = os.environ.get("GLOBAL_MODEL_DIR")
        try:
            with tempfile.TemporaryDirectory() as directory:
                os.environ["GLOBAL_MODEL_DIR"] = directory
                index = pd.bdate_range("2025-01-01", periods=140)
                random = np.random.default_rng(42)
                symbols = ["1001.T", "1002.T", "1003.T", "1004.T"]
                returns = pd.DataFrame(
                    random.normal(0.0005, 0.01, (len(index), len(symbols))),
                    index=index, columns=symbols,
                )
                close = 100 * (1 + returns).cumprod()
                volume = pd.DataFrame(100000.0, index=index, columns=symbols)
                universe = pd.DataFrame({
                    "code": ["1001", "1002", "1003", "1004"],
                    "sector": ["機械", "機械", "食品", "食品"],
                })

                metadata = hybrid_model.train_and_promote(close, volume, universe)
                source = pd.DataFrame({
                    "Close": close["1001.T"], "Volume": volume["1001.T"],
                    "topix_return": close.pct_change().median(axis=1),
                })
                prediction = hybrid_model.predict_for_stock(source, index[-20:], "機械")

                self.assertTrue(metadata["promoted"])
                self.assertTrue(prediction["available"])
                self.assertEqual(len(prediction["holdout_prediction"]), 20)
                self.assertIn("validation_rank_ic", metadata)
                self.assertTrue(Path(directory, f"champion_{hybrid_model.MODEL_VERSION}.pkl").exists())
        finally:
            if previous is None:
                os.environ.pop("GLOBAL_MODEL_DIR", None)
            else:
                os.environ["GLOBAL_MODEL_DIR"] = previous

    def test_bulk_market_data_is_split_and_sector_is_mapped(self):
        columns = pd.MultiIndex.from_product([["Close", "Open"], ["5802.T", "^N225"]])
        downloaded = pd.DataFrame([[100, 200, 99, 198]], columns=columns)
        result = _bulk_symbol_frame(downloaded, "5802.T")
        self.assertEqual(list(result.columns), ["Open", "Close"])
        self.assertEqual(float(result.iloc[0]["Close"]), 100.0)
        self.assertEqual(sector_etf_symbol("電気機器"), "1625.T")

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
                    "generated_date": [now_jst().date().isoformat()] * 2,
                    "positive_factors": [json.dumps(["positive"])] * 2,
                    "risk_factors": [json.dumps([])] * 2,
                })
                atomic_replace_ranking(frame)
                result = read_latest_ranking(limit=1)
                self.assertTrue(result["available"])
                self.assertEqual(result["ranking"][0]["code"], "5802")
                self.assertEqual(result["ranking"][0]["positive_factors"], ["positive"])
                self.assertFalse(result["refresh_allowed"])
                self.assertEqual(read_status()["refresh_block_reason"], "ranking_already_generated_today")
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

    def test_global_prediction_changes_screening_priority(self):
        index = pd.bdate_range("2025-01-01", periods=140)
        close = pd.DataFrame({
            "5802.T": np.linspace(100, 130, len(index)),
            "6501.T": np.linspace(100, 120, len(index)),
        }, index=index)
        volume = pd.DataFrame(1000.0, index=index, columns=close.columns)
        universe = pd.DataFrame({
            "code": ["5802", "6501"], "company": ["A社", "B社"], "sector": ["非鉄金属", "電気機器"]
        })
        global_scores = pd.DataFrame({
            "symbol": ["5802.T", "6501.T"],
            "global_predicted_return": [-0.01, 0.02],
            "global_model_rank": [0.0, 1.0],
        })

        result = screen_prime_universe(universe, close, volume, None, global_scores)

        self.assertEqual(result.iloc[0]["code"], "6501")

    def test_final_ranking_uses_cross_sectional_prediction_quality(self):
        candidates = pd.DataFrame({
            "code": ["1001", "1002", "1003"],
            "expected_value": [0.03, 0.01, -0.01],
            "up_probability_5d": [0.7, 0.55, 0.3],
            "predicted_excess_return": [0.02, 0.0, -0.02],
            "loss_probability": [0.2, 0.45, 0.8],
            "reward_risk_ratio": [2.0, 1.0, 0.5],
            "confidence_score": [80, 60, 30],
            "fundamental_score": [70, 60, 50],
            "screening_score": [70, 80, 90],
        })

        result = rerank_enriched_candidates(candidates)

        self.assertEqual(result["code"].tolist(), ["1001", "1002", "1003"])
        self.assertTrue(result["total_score"].is_monotonic_decreasing)

    def test_stale_running_status_is_recovered_after_worker_disappears(self):
        previous_dir = os.environ.get("PRIME_RANKING_DIR")
        previous_timeout = os.environ.get("PRIME_RANKING_STALE_SECONDS")
        try:
            with tempfile.TemporaryDirectory() as directory:
                os.environ["PRIME_RANKING_DIR"] = directory
                os.environ["PRIME_RANKING_STALE_SECONDS"] = "60"
                old = (now_jst() - pd.Timedelta(minutes=5)).isoformat()
                Path(directory, "analysis_status.json").write_text(json.dumps({
                    "status": "running",
                    "started_at": old,
                    "updated_at": old,
                    "phase": "analyzing_candidates",
                    "progress_percent": 48,
                }), encoding="utf-8")

                status = read_status()

                self.assertEqual(status["status"], "failed")
                self.assertTrue(status["stale_recovered"])
                self.assertTrue(status["refresh_allowed"])
        finally:
            if previous_dir is None:
                os.environ.pop("PRIME_RANKING_DIR", None)
            else:
                os.environ["PRIME_RANKING_DIR"] = previous_dir
            if previous_timeout is None:
                os.environ.pop("PRIME_RANKING_STALE_SECONDS", None)
            else:
                os.environ["PRIME_RANKING_STALE_SECONDS"] = previous_timeout

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
