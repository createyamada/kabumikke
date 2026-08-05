"""TSE Prime universe screening with an atomic latest-only CSV store."""
import io
import json
import os
import re
import threading
from datetime import datetime
from pathlib import Path
from urllib.parse import urljoin

import numpy as np
import pandas as pd

from services import analysis


JPX_LIST_PAGE = "https://www.jpx.co.jp/markets/statistics-equities/misc/01.html"
REQUIRED_COLUMNS = {"rank", "code", "company", "total_score", "analyzed_at"}
_worker_lock = threading.Lock()


def ranking_paths():
    root = Path(os.getenv("PRIME_RANKING_DIR", "data/prime_ranking"))
    return {
        "root": root,
        "latest": root / "prime_ranking_latest.csv",
        "universe": root / "prime_universe_latest.csv",
        "status": root / "analysis_status.json",
        "temporary": root / "work" / "prime_ranking_building.csv",
    }


def atomic_write_text(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def write_status(status, **values):
    payload = {"status": status, **values}
    atomic_write_text(ranking_paths()["status"], json.dumps(payload, ensure_ascii=False, indent=2))
    return payload


def read_status():
    path = ranking_paths()["status"]
    if not path.exists():
        return {"status": "not_started", "latest_csv_exists": ranking_paths()["latest"].exists()}
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["latest_csv_exists"] = ranking_paths()["latest"].exists()
    return payload


def _download_jpx_excel():
    import requests

    response = requests.get(JPX_LIST_PAGE, timeout=30)
    response.raise_for_status()
    links = re.findall(r'href=["\']([^"\']+\.(?:xlsx?|xlsm))(?:\?[^"\']*)?["\']', response.text, re.I)
    if not links:
        raise RuntimeError("JPX listed-company Excel link was not found")
    # ページ内で最初に掲載される最新一覧を使用する。
    excel_response = requests.get(urljoin(JPX_LIST_PAGE, links[0]), timeout=60)
    excel_response.raise_for_status()
    return excel_response.content


def _find_column(frame, keywords):
    for column in frame.columns:
        normalized = str(column).replace(" ", "").lower()
        if any(keyword.lower() in normalized for keyword in keywords):
            return column
    return None


def parse_prime_universe(frame):
    """JPX Excelの日本語・英語列名の双方からプライム内国株式を抽出する。"""
    code_column = _find_column(frame, ("コード", "code"))
    name_column = _find_column(frame, ("銘柄名", "会社名", "name"))
    market_column = _find_column(frame, ("市場・商品区分", "市場区分", "market"))
    sector_column = _find_column(frame, ("33業種区分", "業種区分", "sector"))
    if not code_column or not name_column or not market_column:
        raise ValueError("JPX Excel does not contain required code/name/market columns")
    market = frame[market_column].astype(str)
    prime = frame[market.str.contains("プライム|Prime", case=False, regex=True, na=False)].copy()
    # ETF・REIT等を除き、4桁の普通株式コードに限定する。
    prime["code"] = prime[code_column].astype(str).str.extract(r"(\d{4})", expand=False)
    prime = prime[prime["code"].notna()]
    if market.str.contains("内国株式|Domestic Stock", case=False, regex=True, na=False).any():
        prime = prime[prime[market_column].astype(str).str.contains("内国株式|Domestic Stock", case=False, regex=True, na=False)]
    result = pd.DataFrame({
        "code": prime["code"],
        "company": prime[name_column].astype(str),
        "sector": prime[sector_column].astype(str) if sector_column else "",
        "market": prime[market_column].astype(str),
    }).drop_duplicates("code").sort_values("code")
    if result.empty:
        raise ValueError("no TSE Prime common stocks were found")
    return result.reset_index(drop=True)


def fetch_prime_universe():
    universe = parse_prime_universe(pd.read_excel(io.BytesIO(_download_jpx_excel())))
    path = ranking_paths()["universe"]
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".csv.tmp")
    universe.to_csv(temporary, index=False, encoding="utf-8-sig")
    os.replace(temporary, path)
    return universe


def _extract_field(downloaded, field):
    if isinstance(downloaded.columns, pd.MultiIndex):
        if field in downloaded.columns.get_level_values(0):
            return downloaded[field]
        if field in downloaded.columns.get_level_values(-1):
            return downloaded.xs(field, axis=1, level=-1)
    if field in downloaded.columns:
        return downloaded[[field]]
    return pd.DataFrame(index=downloaded.index)


def download_market_matrices(codes, period="2y", chunk_size=100):
    import yfinance as yf

    closes = []
    volumes = []
    symbols = [f"{code}.T" for code in codes]
    for start in range(0, len(symbols), chunk_size):
        chunk = symbols[start:start + chunk_size]
        downloaded = yf.download(chunk, period=period, auto_adjust=True, progress=False, threads=True)
        closes.append(_extract_field(downloaded, "Close"))
        volumes.append(_extract_field(downloaded, "Volume"))
    close = pd.concat(closes, axis=1).loc[:, lambda x: ~x.columns.duplicated()]
    volume = pd.concat(volumes, axis=1).loc[:, lambda x: ~x.columns.duplicated()]
    topix_download = yf.download("^TOPX", period=period, auto_adjust=True, progress=False)
    topix = _extract_field(topix_download, "Close").iloc[:, 0]
    return close, volume, topix


def screen_prime_universe(universe, close, volume, topix):
    """全銘柄をベクトル演算し、高度分析候補をランキングする。"""
    close = close.ffill()
    volume = volume.reindex_like(close).fillna(0)
    valid = close.notna().sum() >= 120
    close = close.loc[:, valid]
    volume = volume.reindex(columns=close.columns)
    latest = close.iloc[-1]
    return20 = close.pct_change(20).iloc[-1]
    return60 = close.pct_change(60).iloc[-1]
    topix20 = float(topix.ffill().pct_change(20).iloc[-1])
    excess20 = return20 - topix20
    volatility20 = close.pct_change().tail(20).std()
    average_turnover20 = (close * volume).tail(20).mean()
    volume_ratio20 = volume.tail(20).mean() / volume.tail(60).mean().replace(0, np.nan) - 1
    table = pd.DataFrame({
        "symbol": close.columns.astype(str),
        "last_close": latest.values,
        "return_20d": return20.values,
        "return_60d": return60.values,
        "topix_excess_return_20d": excess20.values,
        "volatility_20d": volatility20.values,
        "average_turnover_20d": average_turnover20.values,
        "volume_ratio_20d": volume_ratio20.values,
    })
    table["code"] = table["symbol"].str.extract(r"(\d{4})", expand=False)
    table = table.merge(universe[["code", "company", "sector"]], on="code", how="inner")
    table = table.replace([np.inf, -np.inf], np.nan).dropna(subset=["return_20d", "volatility_20d"])
    table["screening_score"] = 100 * (
        0.25 * table["return_20d"].rank(pct=True)
        + 0.25 * table["topix_excess_return_20d"].rank(pct=True)
        + 0.15 * table["return_60d"].rank(pct=True)
        + 0.10 * table["volume_ratio_20d"].rank(pct=True)
        + 0.15 * table["average_turnover_20d"].rank(pct=True)
        + 0.10 * (1 - table["volatility_20d"].rank(pct=True))
    )
    return table.sort_values("screening_score", ascending=False).reset_index(drop=True)


def _safe_number(value, default=0.0):
    try:
        value = float(value)
        return value if np.isfinite(value) else default
    except (TypeError, ValueError):
        return default


def enrich_candidate(row):
    """既存の単銘柄高度分析を候補銘柄へ適用し、ランキング用の平坦な行へ変換する。"""
    result = analysis.get_prediction(row["code"])
    prediction = result["prediction"]
    horizon5 = prediction.get("horizon_predictions", {}).get("5", {})
    risk = prediction.get("return_risk", {})
    excess = prediction.get("topix_excess_return_prediction", {})
    confidence = prediction.get("confidence", {})
    fundamental = prediction.get("fundamental_analysis", {})
    topology = prediction.get("topological_analysis", {})
    financial_score = fundamental.get("assessment", {}).get("score") if fundamental.get("available") else None
    expected = _safe_number(risk.get("expected_return_after_cost"))
    probability = _safe_number(horizon5.get("up_probability"), 0.5)
    excess_return = _safe_number(excess.get("predicted_excess_return"))
    confidence_score = _safe_number(confidence.get("confidence_score"), 50)
    reward_risk = _safe_number(risk.get("reward_risk_ratio"), 0)
    total_score = np.clip(
        0.30 * _safe_number(row["screening_score"])
        + 20 * np.clip(expected / 0.03, -1, 1)
        + 15 * np.clip((probability - 0.5) / 0.20, -1, 1)
        + 10 * np.clip(excess_return / 0.03, -1, 1)
        + 0.15 * confidence_score
        + 0.10 * (_safe_number(financial_score, 50)),
        0, 100,
    )
    positive = []
    risks = list(confidence.get("risk_reasons", []))
    if expected > 0:
        positive.append("取引コスト控除後の期待値がプラス")
    if probability >= 0.55:
        positive.append("5営業日の上昇確率が55%以上")
    if excess_return > 0:
        positive.append("TOPIX超過収益予測がプラス")
    if topology.get("regime") == "high_topological_complexity":
        risks.append("TDAが高トポロジー複雑度")
    return {
        **row.to_dict(),
        "company": result.get("company") or row["company"],
        "total_score": float(total_score),
        "trade_signal": confidence.get("trade_signal", "判定不能"),
        "predicted_return_5d": horizon5.get("predicted_return"),
        "up_probability_5d": horizon5.get("up_probability"),
        "predicted_excess_return": excess.get("predicted_excess_return"),
        "expected_value": risk.get("expected_return_after_cost"),
        "loss_probability": risk.get("loss_probability"),
        "reward_risk_ratio": risk.get("reward_risk_ratio"),
        "confidence_score": confidence.get("confidence_score"),
        "fundamental_score": financial_score,
        "fundamental_data_coverage": fundamental.get("assessment", {}).get("data_coverage") if fundamental.get("available") else 0.0,
        "topological_regime": topology.get("regime"),
        "positive_factors": json.dumps(positive, ensure_ascii=False),
        "risk_factors": json.dumps(risks, ensure_ascii=False),
    }


def atomic_replace_ranking(frame):
    paths = ranking_paths()
    paths["temporary"].parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(paths["temporary"], index=False, encoding="utf-8-sig")
    verified = pd.read_csv(paths["temporary"], dtype={"code": str})
    missing = REQUIRED_COLUMNS.difference(verified.columns)
    if verified.empty or missing:
        raise ValueError(f"ranking CSV validation failed; missing={sorted(missing)}")
    if verified["code"].duplicated().any() or not verified["total_score"].is_monotonic_decreasing:
        raise ValueError("ranking CSV ordering or uniqueness validation failed")
    paths["latest"].parent.mkdir(parents=True, exist_ok=True)
    os.replace(paths["temporary"], paths["latest"])


def build_prime_ranking(limit=10, shortlist_size=50):
    started = datetime.now().astimezone().isoformat()
    write_status("running", started_at=started, analyzed_count=0, failed_count=0)
    try:
        universe = fetch_prime_universe()
        close, volume, topix = download_market_matrices(universe["code"].tolist())
        screened = screen_prime_universe(universe, close, volume, topix)
        enriched = []
        failures = []
        for _, row in screened.head(shortlist_size).iterrows():
            try:
                enriched.append(enrich_candidate(row))
            except Exception as error:
                failures.append({"code": row["code"], "error": str(error)})
        if not enriched:
            raise RuntimeError("all advanced candidate analyses failed")
        ranking = pd.DataFrame(enriched).sort_values("total_score", ascending=False).reset_index(drop=True)
        ranking.insert(0, "rank", np.arange(1, len(ranking) + 1))
        ranking["analyzed_at"] = datetime.now().astimezone().isoformat()
        # CSVには候補全件を残し、API側でlimitを適用する。
        atomic_replace_ranking(ranking)
        write_status(
            "completed", started_at=started, completed_at=datetime.now().astimezone().isoformat(),
            universe_count=int(len(universe)), screening_count=int(len(screened)),
            analyzed_count=int(len(enriched)), failed_count=int(len(failures)), failures=failures[:20],
        )
        return ranking.head(limit)
    except Exception as error:
        write_status("failed", started_at=started, failed_at=datetime.now().astimezone().isoformat(), error=str(error))
        raise


def start_prime_ranking_refresh(limit=10, shortlist_size=50):
    if not _worker_lock.acquire(blocking=False):
        return {"status": "already_running"}

    def worker():
        try:
            build_prime_ranking(limit=limit, shortlist_size=shortlist_size)
        finally:
            _worker_lock.release()

    thread = threading.Thread(target=worker, name="prime-ranking-refresh", daemon=True)
    thread.start()
    return {"status": "queued", "started_at": datetime.now().astimezone().isoformat()}


def read_latest_ranking(limit=10):
    path = ranking_paths()["latest"]
    if not path.exists():
        return {"available": False, "reason": "ranking_not_generated", "ranking": []}
    frame = pd.read_csv(path, dtype={"code": str}).head(limit)
    for column in ("positive_factors", "risk_factors"):
        if column in frame:
            frame[column] = frame[column].apply(lambda value: json.loads(value) if pd.notna(value) else [])
    frame = frame.replace({np.nan: None})
    status = read_status()
    return {
        "available": True,
        "generated_at": frame["analyzed_at"].iloc[0] if len(frame) else None,
        "source": path.name,
        "universe_count": status.get("universe_count"),
        "analyzed_count": status.get("analyzed_count"),
        "ranking": frame.to_dict(orient="records"),
    }
