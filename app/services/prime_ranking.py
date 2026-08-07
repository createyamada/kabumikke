"""TSE Prime universe screening with an atomic latest-only CSV store."""
import io
import json
import multiprocessing
import os
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from pathlib import Path
from urllib.parse import urljoin
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from services import analysis


JPX_LIST_PAGE = "https://www.jpx.co.jp/markets/statistics-equities/misc/01.html"
REQUIRED_COLUMNS = {"rank", "code", "company", "total_score", "analyzed_at"}
_worker_lock = threading.Lock()
JST = ZoneInfo("Asia/Tokyo")

SECTOR_ETF_RULES = (
    (("水産", "農林", "食品"), "1617.T"),
    (("鉱業", "石油", "石炭"), "1618.T"),
    (("建設",), "1619.T"),
    (("繊維", "パルプ", "紙", "化学"), "1620.T"),
    (("医薬品",), "1621.T"),
    (("ゴム", "輸送用機器"), "1622.T"),
    (("鉄鋼", "非鉄金属", "金属製品"), "1623.T"),
    (("機械",), "1624.T"),
    (("電気機器", "精密機器"), "1625.T"),
    (("情報・通信", "サービス"), "1626.T"),
    (("電気・ガス",), "1627.T"),
    (("陸運", "海運", "空運", "倉庫"), "1628.T"),
    (("卸売",), "1629.T"),
    (("小売",), "1630.T"),
    (("銀行",), "1631.T"),
    (("証券", "保険", "その他金融"), "1632.T"),
    (("不動産",), "1633.T"),
)


def now_jst():
    return datetime.now(JST)


def latest_ranking_date():
    path = ranking_paths()["latest"]
    if not path.exists():
        return None
    try:
        frame = pd.read_csv(path, nrows=1, dtype=str)
        if frame.empty:
            return None
        if "generated_date" in frame:
            return str(frame["generated_date"].iloc[0])
        if "analyzed_at" in frame:
            timestamp = pd.to_datetime(frame["analyzed_at"].iloc[0])
            timestamp = timestamp.tz_localize(JST) if timestamp.tzinfo is None else timestamp.tz_convert(JST)
            return timestamp.date().isoformat()
    except (OSError, ValueError, KeyError, TypeError):
        return None
    return None


def refresh_availability():
    generated_date = latest_ranking_date()
    today = now_jst().date().isoformat()
    allowed = generated_date != today
    return {
        "refresh_allowed": allowed,
        "refresh_block_reason": None if allowed else "ranking_already_generated_today",
        "latest_generated_date": generated_date,
        "today_jst": today,
    }


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


def write_progress(started_at, phase, phase_label, progress_percent, **values):
    """Persist a UI-friendly progress snapshot, including a rolling ETA."""
    now = now_jst()
    try:
        started = datetime.fromisoformat(started_at)
        elapsed = max(0.0, (now - started).total_seconds())
    except (TypeError, ValueError):
        elapsed = 0.0
    progress = float(np.clip(progress_percent, 0, 100))
    remaining = None
    estimated_completion = None
    if 1 <= progress < 100 and elapsed > 0:
        remaining = max(0, round(elapsed * (100 - progress) / progress))
        estimated_completion = (now + timedelta(seconds=remaining)).isoformat()
    return write_status(
        "running",
        started_at=started_at,
        updated_at=now.isoformat(),
        phase=phase,
        phase_label=phase_label,
        progress_percent=round(progress, 1),
        elapsed_seconds=round(elapsed),
        estimated_remaining_seconds=remaining,
        estimated_completion_at=estimated_completion,
        **values,
    )


def read_status():
    path = ranking_paths()["status"]
    if not path.exists():
        return {
            "status": "not_started",
            "latest_csv_exists": ranking_paths()["latest"].exists(),
            **refresh_availability(),
        }
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["latest_csv_exists"] = ranking_paths()["latest"].exists()
    payload.update(refresh_availability())
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


def _extract_field(downloaded, field, symbols=None):
    if isinstance(downloaded.columns, pd.MultiIndex):
        if field in downloaded.columns.get_level_values(0):
            return downloaded[field]
        if field in downloaded.columns.get_level_values(-1):
            return downloaded.xs(field, axis=1, level=-1)
    if field in downloaded.columns:
        result = downloaded[[field]].copy()
        if symbols and len(symbols) == 1:
            result.columns = [symbols[0]]
        return result
    return pd.DataFrame(index=downloaded.index)


def download_market_matrices(codes, period="2y", chunk_size=100, progress_callback=None):
    import yfinance as yf

    closes = []
    volumes = []
    symbols = [f"{code}.T" for code in codes]
    chunks = list(range(0, len(symbols), chunk_size))
    for chunk_index, start in enumerate(chunks, start=1):
        chunk = symbols[start:start + chunk_size]
        downloaded = yf.download(chunk, period=period, auto_adjust=True, progress=False, threads=True)
        closes.append(_extract_field(downloaded, "Close", chunk))
        volumes.append(_extract_field(downloaded, "Volume", chunk))
        if progress_callback:
            progress_callback(chunk_index, len(chunks))
    close = pd.concat(closes, axis=1).loc[:, lambda x: ~x.columns.duplicated()]
    volume = pd.concat(volumes, axis=1).loc[:, lambda x: ~x.columns.duplicated()]
    topix = None
    # Yahoo側でTOPIX指数が取得できない場合はTOPIX連動ETF（1306）へフォールバックする。
    for benchmark_symbol in ("^TOPX", "1306.T"):
        try:
            topix_download = yf.download(benchmark_symbol, period=period, auto_adjust=True, progress=False)
            topix_frame = _extract_field(topix_download, "Close", [benchmark_symbol])
            if not topix_frame.empty and topix_frame.notna().any().any():
                topix = topix_frame.iloc[:, 0].dropna()
                break
        except Exception:
            continue
    return close, volume, topix


def sector_etf_symbol(sector_name):
    name = str(sector_name or "")
    for keywords, symbol in SECTOR_ETF_RULES:
        if any(keyword in name for keyword in keywords):
            return symbol
    return None


def _bulk_symbol_frame(downloaded, symbol):
    if downloaded is None or downloaded.empty:
        return pd.DataFrame()
    if isinstance(downloaded.columns, pd.MultiIndex):
        if symbol in downloaded.columns.get_level_values(0):
            frame = downloaded.xs(symbol, axis=1, level=0)
        elif symbol in downloaded.columns.get_level_values(-1):
            frame = downloaded.xs(symbol, axis=1, level=-1)
        else:
            return pd.DataFrame()
    else:
        frame = downloaded.copy()
    wanted = [column for column in ("Open", "High", "Low", "Close", "Volume") if column in frame]
    return frame[wanted].dropna(how="all").copy() if wanted else pd.DataFrame()


def download_candidate_analysis_data(candidates, period=None):
    """Fetch candidate, common-market and sector histories in one bulk request."""
    import yfinance as yf

    period = period or analysis.HISTORY_PERIOD
    rows = candidates.to_dict(orient="records")
    sector_symbols = {sector_etf_symbol(row.get("sector")) for row in rows}
    sector_symbols.discard(None)
    common_symbols = {"^N225", "^TOPX", "1306.T", "JPY=X", "^DJI", "YM=F"}
    symbols = [f"{row['code']}.T" for row in rows] + sorted(common_symbols | sector_symbols)
    downloaded = yf.download(
        symbols, period=period, auto_adjust=True, progress=False,
        threads=True, group_by="column",
    )
    histories = {symbol: _bulk_symbol_frame(downloaded, symbol) for symbol in symbols}
    topix = histories.get("^TOPX")
    if topix is None or topix.empty:
        topix = histories.get("1306.T", pd.DataFrame())
    result = {}
    for row in rows:
        sector_symbol = sector_etf_symbol(row.get("sector"))
        result[str(row["code"])] = {
            "company": histories.get(f"{row['code']}.T", pd.DataFrame()),
            "nikkei": histories.get("^N225", pd.DataFrame()),
            "topix": topix,
            "jpy": histories.get("JPY=X", pd.DataFrame()),
            "dow": histories.get("^DJI", pd.DataFrame()),
            "mini_dow": histories.get("YM=F", pd.DataFrame()),
            "sector": histories.get(sector_symbol, pd.DataFrame()) if sector_symbol else pd.DataFrame(),
            "sector_symbol": sector_symbol,
            "sector_name": row.get("sector"),
        }
    return result


def screen_prime_universe(universe, close, volume, topix):
    """全銘柄をベクトル演算し、高度分析候補をランキングする。"""
    close = close.ffill()
    if close.empty or len(close) < 61:
        raise ValueError("insufficient stock price data returned by yfinance")
    volume = volume.reindex_like(close).fillna(0)
    valid = close.notna().sum() >= 120
    close = close.loc[:, valid]
    volume = volume.reindex(columns=close.columns)
    latest = close.iloc[-1]
    return20 = close.pct_change(20).iloc[-1]
    return60 = close.pct_change(60).iloc[-1]
    if topix is not None and len(topix.dropna()) >= 21:
        topix20 = float(topix.ffill().pct_change(20).dropna().iloc[-1])
        benchmark_source = "TOPIX_or_1306"
    else:
        # 指数・ETFの両方が取れない場合でも停止せず、分析対象銘柄の中央値を市場代理値にする。
        topix20 = float(return20.median())
        benchmark_source = "prime_universe_median_fallback"
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
        "market_benchmark_source": benchmark_source,
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


def enrich_candidate(row, preloaded=None):
    """既存の単銘柄高度分析を候補銘柄へ適用し、ランキング用の平坦な行へ変換する。"""
    result = analysis.get_prediction(
        row["code"], preloaded=preloaded, company_name=row.get("company"),
    )
    prediction = result["prediction"]
    horizon5 = prediction.get("horizon_predictions", {}).get("5", {})
    risk = prediction.get("return_risk", {})
    excess = prediction.get("topix_excess_return_prediction", {})
    confidence = prediction.get("confidence", {})
    fundamental = prediction.get("fundamental_analysis", {})
    topology = prediction.get("topological_analysis") or {}
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
    if not refresh_availability()["refresh_allowed"]:
        raise RuntimeError("ranking_already_generated_today")
    started = now_jst().isoformat()
    common = {"analyzed_count": 0, "processed_count": 0, "failed_count": 0, "total_count": shortlist_size}
    write_progress(started, "fetching_universe", "東証プライム銘柄一覧を取得中", 2, **common)
    try:
        universe = fetch_prime_universe()
        common["universe_count"] = int(len(universe))
        write_progress(started, "downloading_prices", "全銘柄の株価・出来高を取得中", 10, **common)

        def market_progress(completed_chunks, total_chunks):
            progress = 10 + 25 * completed_chunks / max(total_chunks, 1)
            write_progress(
                started, "downloading_prices", "全銘柄の株価・出来高を取得中", progress,
                market_chunks_completed=completed_chunks, market_chunks_total=total_chunks, **common,
            )

        close, volume, topix = download_market_matrices(
            universe["code"].tolist(), progress_callback=market_progress,
        )
        write_progress(started, "screening", "テクニカル指標で候補を抽出中", 36, **common)
        screened = screen_prime_universe(universe, close, volume, topix)
        target_count = min(shortlist_size, len(screened))
        candidates = screened.head(shortlist_size).copy()
        common.update(screening_count=int(len(screened)), total_count=int(target_count))
        write_progress(started, "preloading_candidate_data", "候補銘柄の共通データを一括取得中", 40, **common)
        preloaded_by_code = download_candidate_analysis_data(candidates)
        write_progress(started, "analyzing_candidates", "候補銘柄を並列分析中", 48, **common)
        enriched = []
        failures = []
        worker_count = min(4, max(1, int(os.getenv("PRIME_RANKING_WORKERS", "3"))))
        with ThreadPoolExecutor(max_workers=worker_count, thread_name_prefix="prime-analysis") as executor:
            futures = {
                executor.submit(
                    enrich_candidate, row, preloaded_by_code.get(str(row["code"])),
                ): row
                for _, row in candidates.iterrows()
            }
            for processed, future in enumerate(as_completed(futures), start=1):
                row = futures[future]
                try:
                    enriched.append(future.result())
                except Exception as error:
                    failures.append({"code": row["code"], "error": str(error)})
                common.update(
                    analyzed_count=len(enriched), processed_count=processed,
                    failed_count=len(failures), current_code=str(row["code"]),
                    current_company=str(row["company"]),
                )
                write_progress(
                    started, "analyzing_candidates", f"候補銘柄を{worker_count}並列で分析中",
                    48 + 47 * processed / max(target_count, 1),
                    worker_count=worker_count, **common,
                )
        if not enriched:
            raise RuntimeError("all advanced candidate analyses failed")
        write_progress(started, "saving", "ランキングCSVを検証・保存中", 97, **common)
        ranking = pd.DataFrame(enriched).sort_values("total_score", ascending=False).reset_index(drop=True)
        ranking.insert(0, "rank", np.arange(1, len(ranking) + 1))
        completed = now_jst()
        ranking["generated_date"] = completed.date().isoformat()
        ranking["analyzed_at"] = completed.isoformat()
        # CSVには候補全件を残し、API側でlimitを適用する。
        atomic_replace_ranking(ranking)
        write_status(
            "completed", started_at=started, completed_at=completed.isoformat(),
            updated_at=completed.isoformat(), phase="completed", phase_label="ランキング生成完了",
            progress_percent=100.0, elapsed_seconds=round((completed - datetime.fromisoformat(started)).total_seconds()),
            estimated_remaining_seconds=0, estimated_completion_at=completed.isoformat(),
            universe_count=int(len(universe)), screening_count=int(len(screened)),
            total_count=int(target_count), processed_count=int(target_count),
            analyzed_count=int(len(enriched)), failed_count=int(len(failures)), failures=failures[:20],
        )
        return ranking.head(limit)
    except Exception as error:
        write_status("failed", started_at=started, failed_at=now_jst().isoformat(), error=str(error))
        raise


def _ranking_process_worker(limit, shortlist_size):
    build_prime_ranking(limit=limit, shortlist_size=shortlist_size)


def start_prime_ranking_refresh(limit=10, shortlist_size=50):
    availability = refresh_availability()
    if not availability["refresh_allowed"]:
        return {"status": "already_generated_today", **availability}
    if not _worker_lock.acquire(blocking=False):
        return {"status": "already_running"}

    queued_at = now_jst().isoformat()
    write_status(
        "queued", started_at=queued_at, updated_at=queued_at,
        phase="queued", phase_label="ランキング処理を開始しています",
        progress_percent=0.0, analyzed_count=0, processed_count=0,
        failed_count=0, total_count=shortlist_size,
    )
    process = None
    try:
        # A separate process keeps CPU-heavy ranking work from blocking individual analysis.
        if multiprocessing.current_process().daemon:
            raise RuntimeError("process spawning is unavailable from a daemon worker")
        process = multiprocessing.Process(
            target=_ranking_process_worker,
            args=(limit, shortlist_size),
            name="prime-ranking-refresh",
            daemon=True,
        )
        process.start()
    except Exception:
        # Some application servers run daemon workers that cannot create children.
        # A background thread still avoids any application-wide request lock.
        def thread_worker():
            try:
                build_prime_ranking(limit=limit, shortlist_size=shortlist_size)
            finally:
                _worker_lock.release()

        threading.Thread(target=thread_worker, name="prime-ranking-refresh", daemon=True).start()
        return {**read_status(), **refresh_availability()}

    def monitor():
        process.join()
        _worker_lock.release()
        if process.exitcode and read_status().get("status") not in {"failed", "completed"}:
            write_status(
                "failed", started_at=queued_at, failed_at=now_jst().isoformat(),
                phase="failed", phase_label="ランキング生成に失敗しました",
                progress_percent=0.0, error=f"ranking worker exited with code {process.exitcode}",
            )

    threading.Thread(target=monitor, name="prime-ranking-monitor", daemon=True).start()
    return {**read_status(), **refresh_availability()}


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
        **refresh_availability(),
        "ranking": frame.to_dict(orient="records"),
    }
