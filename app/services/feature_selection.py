"""Auditable, staged feature selection based on permutation importance."""
import json
import os
import statistics
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from services import database


NAMESPACE = "feature_selection_events"
MODEL_VERSION = "feature-selection-v1"


def feature_group(name):
    name = str(name)
    rules = (
        ("market", ("dow_", "nasdaq_", "sp500_", "topix_", "nikkei_", "usd_jpy", "vix_")),
        ("fundamental", ("per", "pbr", "roe", "roa", "eps", "dividend", "market_cap", "edinet_")),
        ("volume", ("volume", "obv", "vwap", "mfi")),
        ("volatility", ("volatility", "atr", "bollinger", "true_range")),
        ("momentum", ("return_", "rsi", "roc", "stochastic", "momentum", "cci", "adx", "di14")),
        ("trend", ("sma", "ema", "macd", "ichimoku", "tenkan", "kijun", "cloud", "trend")),
        ("pattern", ("pattern_", "elliott", "fibonacci", "support", "resistance")),
        ("calendar", ("day_of_", "month", "quarter", "is_month")),
    )
    lowered = name.lower()
    for group, tokens in rules:
        if any(token in lowered for token in tokens):
            return group
    return "other"


def _root():
    return Path(os.getenv("FEATURE_SELECTION_DIR", "data/feature_selection"))


def _event_path(key):
    return _root() / f"{key}.json"


def _safe_key(value):
    return "".join(character if character.isalnum() or character in "-_." else "_" for character in str(value))


def record_evaluation(code, sector, model, market_date, horizon, importances,
                      shadow_features=None, shadow_noninferior=None,
                      analysis_version=MODEL_VERSION, regime=None):
    """Persist exactly one evaluation per stock/date/model/horizon/version."""
    key = _safe_key(f"{market_date}__{code or 'unknown'}__{model}__{horizon}__{analysis_version}")
    event = {
        "event_key": key,
        "code": str(code or ""),
        "sector": str(sector or "unknown"),
        "model": str(model),
        "market_date": str(market_date),
        "horizon": int(horizon),
        "regime": str(regime or "unknown"),
        "analysis_version": str(analysis_version),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "importances": importances,
        "shadow_features": sorted(set(shadow_features or [])),
        "shadow_noninferior": shadow_noninferior,
    }
    if database.put_json(NAMESPACE, key, event):
        return True
    path = _event_path(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(event, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    return True


def list_events(limit=10000):
    events = {event.get("event_key"): event for event in database.list_json(NAMESPACE, limit=limit)}
    root = _root()
    if root.exists():
        for path in sorted(root.glob("*.json"), reverse=True):
            try:
                event = json.loads(path.read_text(encoding="utf-8"))
                events.setdefault(event.get("event_key", path.stem), event)
            except (OSError, ValueError):
                continue
    return sorted(events.values(), key=lambda event: event.get("market_date", ""), reverse=True)[:limit]


def _status(row):
    enough_scope = row["distinct_stocks"] >= 10 and row["distinct_sectors"] >= 3
    candidate = (
        row["evaluated_count"] >= 30 and enough_scope
        and row["exclusion_rate"] >= 0.70 and row["mean_importance"] <= 0
        and row["recent_mean_importance"] <= 0
    )
    if not candidate:
        return "active" if row["evaluated_count"] < 10 else "monitoring"
    if row["shadow_evaluations"] >= 20 and row["shadow_noninferior_rate"] >= 0.80:
        return "removed"
    if row["shadow_evaluations"] >= 5 and row["shadow_noninferior_rate"] >= 0.80:
        return "shadow_excluded"
    return "exclusion_candidate"


def aggregate_stats(code=None, sector=None, model=None, horizon=None, regime=None):
    buckets = defaultdict(lambda: {"values": [], "recent": [], "stocks": set(), "sectors": set(),
                                   "dates": [], "shadow": []})
    events = list_events()
    filtered = [event for event in events if
                (not code or event.get("code") == str(code)) and
                (not sector or event.get("sector") == str(sector)) and
                (not model or event.get("model") == str(model)) and
                (horizon is None or int(event.get("horizon", 1)) == int(horizon)) and
                (not regime or event.get("regime", "unknown") == str(regime))]
    for event in filtered:
        shadow_set = set(event.get("shadow_features") or [])
        for item in event.get("importances") or []:
            feature = item.get("feature")
            if not feature:
                continue
            value = float(item.get("mse_increase", 0))
            bucket = buckets[feature]
            bucket["values"].append(value)
            bucket["stocks"].add(event.get("code") or "unknown")
            bucket["sectors"].add(event.get("sector") or "unknown")
            bucket["dates"].append(event.get("market_date", ""))
            if feature in shadow_set and event.get("shadow_noninferior") is not None:
                bucket["shadow"].append(bool(event["shadow_noninferior"]))
    rows = []
    for feature, bucket in buckets.items():
        values = bucket["values"]
        recent_values = values[:min(10, len(values))]
        excluded = sum(value <= 0 for value in values)
        shadow_count = len(bucket["shadow"])
        row = {
            "feature": feature,
            "group": feature_group(feature),
            "evaluated_count": len(values),
            "exclusion_count": excluded,
            "exclusion_rate": excluded / len(values),
            "mean_importance": statistics.fmean(values),
            "median_importance": statistics.median(values),
            "recent_mean_importance": statistics.fmean(recent_values),
            "distinct_stocks": len(bucket["stocks"]),
            "distinct_sectors": len(bucket["sectors"]),
            "latest_date": max(bucket["dates"]),
            "shadow_evaluations": shadow_count,
            "shadow_noninferior_rate": sum(bucket["shadow"]) / shadow_count if shadow_count else 0.0,
        }
        row["status"] = _status(row)
        rows.append(row)
    return sorted(rows, key=lambda row: (-row["exclusion_rate"], -row["evaluated_count"], row["feature"]))


def get_removed_features(**scope):
    return [row["feature"] for row in aggregate_stats(**scope) if row["status"] == "removed"]


def get_shadow_candidates(**scope):
    return [row["feature"] for row in aggregate_stats(**scope)
            if row["status"] in {"exclusion_candidate", "shadow_excluded"}]


def dashboard(status=None, minimum_evaluations=0, **scope):
    rows = aggregate_stats(**scope)
    rows = [row for row in rows if row["evaluated_count"] >= minimum_evaluations and
            (not status or row["status"] == status)]
    counts = defaultdict(int)
    for row in rows:
        counts[row["status"]] += 1
    return {"features": rows, "summary": {"feature_count": len(rows), "status_counts": dict(counts),
                                           "event_count": len(list_events())}, "criteria": {
        "candidate": "30 evaluations, 10 stocks, 3 sectors, exclusion rate >= 70%, mean and recent importance <= 0",
        "removed": "20 shadow evaluations and >= 80% non-inferiority",
    }}
