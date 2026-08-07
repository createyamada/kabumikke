import sys
import types
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


APP_DIR = Path(__file__).resolve().parents[1] / "app"
sys.path.insert(0, str(APP_DIR))

# 分析の純粋な計算部分だけをテストするため、外部API依存を軽量な代替にする。
fastapi = types.ModuleType("fastapi")


class HTTPException(Exception):
    def __init__(self, status_code, detail):
        self.status_code = status_code
        self.detail = detail


fastapi.HTTPException = HTTPException
sys.modules.setdefault("fastapi", fastapi)
sys.modules.setdefault("yfinance", types.ModuleType("yfinance"))

ripser_module = types.ModuleType("ripser")
ripser_module.ripser = lambda points, maxdim=1: {
    "dgms": [
        np.array([[0.0, 0.4], [0.0, 0.8], [0.0, np.inf]]),
        np.array([[0.2, 0.5], [0.3, 0.9]]),
    ]
}
sys.modules.setdefault("ripser", ripser_module)

sklearn = types.ModuleType("sklearn")
linear_model = types.ModuleType("sklearn.linear_model")
model_selection = types.ModuleType("sklearn.model_selection")
metrics = types.ModuleType("sklearn.metrics")
ensemble = types.ModuleType("sklearn.ensemble")
pipeline = types.ModuleType("sklearn.pipeline")
preprocessing = types.ModuleType("sklearn.preprocessing")
isotonic = types.ModuleType("sklearn.isotonic")


class LinearRegression:
    def __init__(self, *args, **kwargs):
        pass

    def fit(self, features, target):
        matrix = np.column_stack([np.ones(len(features)), np.asarray(features)])
        self.coefficients = np.linalg.lstsq(matrix, np.asarray(target), rcond=None)[0]
        return self

    def predict(self, features):
        matrix = np.column_stack([np.ones(len(features)), np.asarray(features)])
        return matrix @ self.coefficients

    def predict_proba(self, features):
        score = np.clip(self.predict(features), -20, 20)
        probability = 1 / (1 + np.exp(-score))
        return np.column_stack([1 - probability, probability])


class IsotonicRegression:
    def __init__(self, *args, **kwargs):
        pass

    def fit(self, values, labels):
        self.minimum = float(np.min(values))
        self.maximum = float(np.max(values))
        return self

    def predict(self, values):
        return np.clip(np.asarray(values, dtype=float), 0, 1)


class TimeSeriesSplit:
    def __init__(self, n_splits, gap=0):
        self.n_splits = n_splits
        self.gap = gap

    def split(self, features):
        fold_size = len(features) // (self.n_splits + 1)
        for fold in range(self.n_splits):
            train_end = fold_size * (fold + 1)
            valid_end = min(train_end + fold_size, len(features))
            purged_train_end = max(1, train_end - self.gap)
            yield np.arange(purged_train_end), np.arange(train_end, valid_end)


linear_model.LinearRegression = LinearRegression
linear_model.Ridge = LinearRegression
linear_model.LogisticRegression = LinearRegression
ensemble.HistGradientBoostingRegressor = LinearRegression
model_selection.TimeSeriesSplit = TimeSeriesSplit
metrics.mean_squared_error = lambda actual, predicted: np.mean(
    (np.asarray(actual) - np.asarray(predicted)) ** 2
)
pipeline.make_pipeline = lambda *steps: steps[-1]
preprocessing.StandardScaler = object
isotonic.IsotonicRegression = IsotonicRegression
sys.modules.setdefault("sklearn", sklearn)
sys.modules.setdefault("sklearn.linear_model", linear_model)
sys.modules.setdefault("sklearn.model_selection", model_selection)
sys.modules.setdefault("sklearn.metrics", metrics)
sys.modules.setdefault("sklearn.ensemble", ensemble)
sys.modules.setdefault("sklearn.pipeline", pipeline)
sys.modules.setdefault("sklearn.preprocessing", preprocessing)
sys.modules.setdefault("sklearn.isotonic", isotonic)

from library import config, format  # noqa: E402
from services.analysis import (  # noqa: E402
    analyze_topology,
    create_delay_embedding,
    fetch_history,
    get_next_weekday,
    price_predict,
)


class StockAnalysisTest(unittest.TestCase):
    def make_data(self, rows=320):
        index = pd.bdate_range("2024-01-01", periods=rows)
        close = pd.Series(100 + np.arange(rows) * 0.2, index=index)
        data = pd.DataFrame(index=index)
        for offset, column in enumerate(config.EXPLANATORY_VARIABLES_ANALYSIS):
            data[column] = close + offset * 0.01
        data["Close"] = close
        data["Close_next"] = close.shift(-1)
        return data

    def test_division_preserves_order_and_provides_full_training_set(self):
        divided = format.get_divided_data(self.make_data())

        self.assertLess(divided["X_train"].index.max(), divided["X_test"].index.min())
        self.assertEqual(len(divided["X_all"]), 319)
        self.assertEqual(list(divided["last_data"].index), config.EXPLANATORY_VARIABLES_ANALYSIS)
        expected_return = 0.2 / 100
        self.assertAlmostEqual(divided["Y_all"].iloc[0], expected_return)

    def test_prediction_retrains_and_returns_comparison_metrics(self):
        divided = format.get_divided_data(self.make_data())
        result = price_predict(divided)

        self.assertIn("baseline_return_rmse", result["metrics"])
        self.assertIn("directional_accuracy", result["metrics"])
        self.assertEqual(result["score"], result["metrics"]["rmse"])
        self.assertGreater(result["metrics"]["training_samples"], result["metrics"]["test_samples"])
        self.assertIn(result["selected_model"], result["model_comparison"])
        self.assertIn("strategy_return", result["backtest"])
        self.assertIn("up_probability", result)
        self.assertIn("direction_classifier", result)
        self.assertIn("return_risk", result)
        self.assertIn("topix_excess_return_prediction", result)
        self.assertTrue(result["topix_excess_return_prediction"]["available"])
        self.assertIn("industry_relative_strength", result)
        self.assertIn("technical_analysis", result)
        self.assertTrue(result["technical_analysis"]["available"])
        self.assertGreaterEqual(result["up_probability"], 0.0)
        self.assertLessEqual(result["up_probability"], 1.0)
        self.assertEqual(set(result["horizon_predictions"]), {"1", "5", "20"})
        self.assertIn(result["confidence"]["confidence_level"], {"高", "中", "低"})
        self.assertIn(result["confidence"]["trade_signal"], {"候補", "監視", "見送り"})
        self.assertIn("holdout_to_walk_forward_rmse_ratio", result["confidence"])
        self.assertEqual(result["confidence"]["score_type"], "heuristic_analysis_health")
        self.assertFalse(result["confidence"]["statistical_confidence"])
        self.assertIn("brier_score", result["probability_evaluation"])
        self.assertIn("actual_coverage", result["interval_evaluation"])
        self.assertEqual(
            result["prediction_interval"]["method"],
            "adaptive_conformal_asymmetric_residual",
        )
        self.assertEqual(result["backtest"]["execution_lag_business_days"], 1)
        self.assertIn("sortino_ratio", result["backtest"])
        self.assertLess(result["prediction_interval"]["lower_price"], result["prediction_interval"]["upper_price"])
        self.assertIsNone(result["topological_analysis"])
        self.assertIsNone(result["topological_analysis_multi_window"])
        self.assertEqual(result["data_quality"]["us_market_lag_business_days"], 1)

    def test_delay_embedding_has_requested_dimension(self):
        points = create_delay_embedding(np.arange(50), dimension=3, delay=2)

        self.assertEqual(points.shape, (46, 3))
        self.assertTrue(np.allclose(points.mean(axis=0), 0.0))

    def test_advanced_technical_features_exist_without_future_leakage(self):
        index = pd.bdate_range("2024-01-01", periods=320)
        base = 100 + np.arange(len(index)) * 0.1 + np.sin(np.arange(len(index)) / 5)

        def build(last_close_adjustment=0):
            close = base.copy()
            close[-1] += last_close_adjustment
            frame = pd.DataFrame({
                "Open": close - 0.2, "High": close + 1.0, "Low": close - 1.0,
                "Close": close, "Volume": 100000 + np.arange(len(index)) * 10,
                "nikkei_open": close, "nikkei_close": close,
                "topix_open": close, "topix_close": close,
                "dow_open": close, "dow_close": close,
                "jpy_open": close, "jpy_close": close,
            }, index=index)
            return format.merge_all_company_info([frame])

        original = build()
        changed_future = build(last_close_adjustment=100)
        self.assertFalse(original[config.EXPLANATORY_VARIABLES_ANALYSIS].iloc[-1].isna().any())
        for column in (
            "sma5_sma25_gap", "golden_cross", "distance_from_high252",
            "candle_body_ratio", "stochastic_k", "adx14", "cci20", "mfi14",
            "vwap20_gap", "ichimoku_cloud_position", "volume_profile_poc_gap",
        ):
            self.assertIn(column, original.columns)
            self.assertAlmostEqual(original[column].iloc[-2], changed_future[column].iloc[-2])

    def test_next_weekday_skips_weekend(self):
        self.assertEqual(get_next_weekday("2026-08-07"), "2026-08-10")

    def test_market_data_fetch_retries_once(self):
        expected = pd.DataFrame({"Close": [100.0]})

        class TemporaryFailureTicker:
            attempts = 0

            def history(self, **kwargs):
                self.attempts += 1
                if self.attempts == 1:
                    raise RuntimeError("temporary failure")
                return expected

        ticker = TemporaryFailureTicker()
        actual = fetch_history(ticker, "test")

        self.assertEqual(ticker.attempts, 2)
        self.assertTrue(actual.equals(expected))


if __name__ == "__main__":
    unittest.main()
