"""Persisted market-wide model with sector residual corrections.

The global model learns patterns shared by TSE Prime stocks.  It never replaces
the stock-specific model directly; analysis.py blends both models according to
their out-of-sample error for the requested stock.
"""
import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from services import database


MODEL_VERSION = "global-v1"
FEATURES = [
    "return_1d", "return_5d", "return_20d", "volatility_20",
    "sma20_gap", "volume_ratio_20", "market_return_1d", "market_excess_20",
]


def _model_path():
    return Path(os.getenv("GLOBAL_MODEL_DIR", ".cache/global_models")) / f"champion_{MODEL_VERSION}.pkl"


def _candidate_models():
    return {
        "ridge": lambda: make_pipeline(StandardScaler(), Ridge(alpha=10.0)),
        "gradient_boosting": lambda: HistGradientBoostingRegressor(
            learning_rate=0.05, max_iter=150, max_leaf_nodes=15,
            l2_regularization=1.0, random_state=42,
        ),
    }


def _market_series(close, topix=None):
    if topix is not None and len(pd.Series(topix).dropna()) >= 30:
        return pd.Series(topix).reindex(close.index).ffill().pct_change()
    return close.pct_change().median(axis=1)


def _stack_frame(frame):
    try:
        return frame.stack(future_stack=True)
    except TypeError:  # pandas < 2.1
        return frame.stack(dropna=False)


def build_training_panel(close, volume, universe, topix=None):
    """Create point-in-time panel features and a next-session return target."""
    close = close.sort_index().ffill()
    volume = volume.reindex_like(close).fillna(0)
    market_return = _market_series(close, topix)
    returns = close.pct_change()
    feature_matrices = {
        "return_1d": returns,
        "return_5d": close.pct_change(5),
        "return_20d": close.pct_change(20),
        "volatility_20": returns.rolling(20).std(),
        "sma20_gap": close / close.rolling(20).mean() - 1,
        "volume_ratio_20": volume / volume.rolling(20).mean().replace(0, np.nan) - 1,
        "market_return_1d": pd.DataFrame(
            np.repeat(market_return.to_numpy()[:, None], len(close.columns), axis=1),
            index=close.index, columns=close.columns,
        ),
        "market_excess_20": close.pct_change(20).sub(
            (1 + market_return).rolling(20).apply(np.prod, raw=True) - 1, axis=0,
        ),
        "target": close.shift(-1) / close - 1,
    }
    panel = pd.concat(
        {name: _stack_frame(frame) for name, frame in feature_matrices.items()}, axis=1,
    ).replace([np.inf, -np.inf], np.nan).dropna()
    panel.index.names = ["date", "symbol"]
    sector_by_code = universe.assign(code=universe["code"].astype(str).str.zfill(4)).set_index("code")["sector"].to_dict()
    codes = panel.index.get_level_values("symbol").astype(str).str.extract(r"(\d{4})", expand=False)
    panel["sector"] = [str(sector_by_code.get(code, "unknown")) for code in codes]
    return panel


def _cap_training_rows(frame):
    maximum = max(10000, int(os.getenv("GLOBAL_MODEL_MAX_TRAINING_ROWS", "250000")))
    if len(frame) <= maximum:
        return frame
    # Recent observations are more relevant under market drift; retain a small
    # deterministic historical sample as protection against one-regime fitting.
    recent = frame.tail(int(maximum * 0.8))
    older = frame.iloc[:-len(recent)].sample(n=maximum - len(recent), random_state=42)
    return pd.concat([older, recent]).sort_index()


def load_champion():
    payload = database.get_bytes("model_registry", f"champion:{MODEL_VERSION}")
    if payload is not None:
        try:
            return pickle.loads(payload)
        except (ValueError, EOFError, pickle.UnpicklingError):
            pass
    try:
        with _model_path().open("rb") as stream:
            return pickle.load(stream)
    except (OSError, ValueError, EOFError, pickle.UnpicklingError):
        return None


def resolve_sector(code, fallback=None):
    """Resolve the same JPX sector label used while training the global model."""
    if code:
        universe = database.get_json("prime_ranking", "universe")
        for record in (universe or {}).get("records", []):
            if str(record.get("code", "")).zfill(4) == str(code).zfill(4):
                return record.get("sector") or fallback
    return fallback


def _save_champion(artifact):
    serialized = pickle.dumps(artifact)
    if database.put_bytes("model_registry", f"champion:{MODEL_VERSION}", serialized):
        database.put_json("model_registry_index", "global_champion", artifact["metadata"])
        return
    path = _model_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(serialized)
    os.replace(temporary, path)


def train_and_promote(close, volume, universe, topix=None, market_date=None):
    """Train challengers and promote only when validation RMSE improves."""
    panel = build_training_panel(close, volume, universe, topix)
    dates = panel.index.get_level_values("date").unique().sort_values()
    if len(dates) < 80:
        raise ValueError("global model requires at least 80 market sessions")
    validation_dates = dates[-max(20, len(dates) // 5):]
    # The target at date t uses close(t+1). Purge the session immediately before
    # validation so its label cannot reach into the validation period.
    earlier_dates = dates[dates < validation_dates[0]]
    training_dates = earlier_dates[:-1]
    train = _cap_training_rows(panel[panel.index.get_level_values("date").isin(training_dates)])
    valid = panel[panel.index.get_level_values("date").isin(validation_dates)]

    comparisons = {}
    fitted = {}
    for name, factory in _candidate_models().items():
        model = factory()
        model.fit(train[FEATURES], train["target"])
        prediction = model.predict(valid[FEATURES])
        comparisons[name] = float(np.sqrt(mean_squared_error(valid["target"], prediction)))
        fitted[name] = (model, prediction)
    selected = min(comparisons, key=comparisons.get)
    validation_prediction = fitted[selected][1]
    validation_rank_ic = pd.Series(validation_prediction).corr(
        pd.Series(valid["target"].to_numpy()), method="spearman"
    )
    residual = valid["target"].to_numpy() - validation_prediction
    sector_frame = pd.DataFrame({"sector": valid["sector"].to_numpy(), "residual": residual})
    sector_corrections = sector_frame.groupby("sector")["residual"].agg(
        lambda values: float(values.mean()) if len(values) >= 20 else 0.0
    ).to_dict()

    final_data = _cap_training_rows(panel)
    final_model = _candidate_models()[selected]()
    final_model.fit(final_data[FEATURES], final_data["target"])
    candidate_rmse = comparisons[selected]
    previous = load_champion()
    previous_rmse = previous.get("metadata", {}).get("validation_rmse") if previous else None
    promoted = previous_rmse is None or candidate_rmse < float(previous_rmse)
    metadata = {
        "model_version": MODEL_VERSION,
        "market_date": market_date or str(dates[-1].date()),
        "selected_model": selected,
        "validation_rmse": candidate_rmse,
        "validation_rank_ic": float(validation_rank_ic) if pd.notna(validation_rank_ic) else 0.0,
        "model_comparison": comparisons,
        "training_rows": int(len(final_data)),
        "validation_rows": int(len(valid)),
        "stock_count": int(panel.index.get_level_values("symbol").nunique()),
        "sector_count": int(panel["sector"].nunique()),
        "promoted": promoted,
        "previous_validation_rmse": previous_rmse,
    }
    artifact = {
        "model": final_model,
        "features": FEATURES,
        "sector_corrections": sector_corrections,
        "metadata": metadata,
    }
    database.put_bytes(
        "model_registry_candidates", f"{metadata['market_date']}:{MODEL_VERSION}", pickle.dumps(artifact),
    )
    if promoted:
        _save_champion(artifact)
    return metadata


def score_universe(close, volume, topix=None):
    """Score every stock using the persisted champion at the latest market date."""
    artifact = load_champion()
    if not artifact:
        return pd.DataFrame(columns=["symbol", "global_predicted_return"])
    close = close.sort_index().ffill()
    volume = volume.reindex_like(close).fillna(0)
    returns = close.pct_change()
    market_return = _market_series(close, topix)
    market_20 = (1 + market_return).rolling(20).apply(np.prod, raw=True) - 1
    latest = pd.DataFrame({
        "return_1d": returns.iloc[-1],
        "return_5d": close.pct_change(5).iloc[-1],
        "return_20d": close.pct_change(20).iloc[-1],
        "volatility_20": returns.rolling(20).std().iloc[-1],
        "sma20_gap": (close / close.rolling(20).mean() - 1).iloc[-1],
        "volume_ratio_20": (volume / volume.rolling(20).mean().replace(0, np.nan) - 1).iloc[-1],
        "market_return_1d": float(market_return.iloc[-1]),
        "market_excess_20": close.pct_change(20).iloc[-1] - float(market_20.iloc[-1]),
    }).replace([np.inf, -np.inf], np.nan).dropna()
    if latest.empty:
        return pd.DataFrame(columns=["symbol", "global_predicted_return"])
    prediction = artifact["model"].predict(latest[artifact["features"]])
    return pd.DataFrame({
        "symbol": latest.index.astype(str),
        "global_predicted_return": prediction,
        "global_model_rank": pd.Series(prediction).rank(pct=True).to_numpy(),
    })


def features_from_stock_data(source):
    close = source["Close"]
    returns = close.pct_change()
    market_return = source.get("topix_return", source.get("nikkei_return"))
    if market_return is None:
        market_return = pd.Series(0.0, index=source.index)
    volume = source["Volume"]
    return pd.DataFrame({
        "return_1d": returns,
        "return_5d": close.pct_change(5),
        "return_20d": close.pct_change(20),
        "volatility_20": returns.rolling(20).std(),
        "sma20_gap": close / close.rolling(20).mean() - 1,
        "volume_ratio_20": volume / volume.rolling(20).mean().replace(0, np.nan) - 1,
        "market_return_1d": market_return,
        "market_excess_20": close.pct_change(20) - (
            (1 + market_return).rolling(20).apply(np.prod, raw=True) - 1
        ),
    }).replace([np.inf, -np.inf], np.nan)


def predict_for_stock(source, dates, sector_name=None):
    artifact = load_champion()
    if not artifact:
        return {"available": False, "reason": "global_model_not_trained"}
    features = features_from_stock_data(source)
    wanted = pd.DatetimeIndex(dates)
    holdout = features.reindex(wanted)
    if holdout.isna().any().any() or features.iloc[-1].isna().any():
        return {"available": False, "reason": "global_features_unavailable"}
    correction = float(artifact.get("sector_corrections", {}).get(str(sector_name), 0.0))
    return {
        "available": True,
        # Sector correction was calibrated after champion validation. Keep it out
        # of historical holdout scoring, but apply it to the genuinely future row.
        "holdout_prediction": artifact["model"].predict(holdout[artifact["features"]]),
        "latest_prediction": float(
            artifact["model"].predict(features.iloc[-1:][artifact["features"]])[0] + correction
        ),
        "sector_correction": correction,
        "metadata": artifact["metadata"],
    }
