"""EDINET API v2 client and point-in-time fundamental analysis."""
import io
import json
import os
import zipfile
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd


BASE_URL = "https://api.edinet-fsa.go.jp/api/v2"
METRIC_ALIASES = {
    "revenue": ("Revenue", "NetSales", "OperatingRevenue", "売上高", "営業収益"),
    "operating_income": ("OperatingIncome", "OperatingProfit", "営業利益"),
    "ordinary_income": ("OrdinaryIncome", "OrdinaryProfit", "経常利益"),
    "net_income": ("ProfitLossAttributableToOwnersOfParent", "NetIncome", "当期純利益"),
    "total_assets": ("Assets", "TotalAssets", "資産合計"),
    "equity": ("Equity", "NetAssets", "純資産", "自己資本"),
    "operating_cash_flow": ("CashFlowsFromUsedInOperatingActivities", "営業活動によるキャッシュ・フロー"),
    "investing_cash_flow": ("CashFlowsFromUsedInInvestingActivities", "投資活動によるキャッシュ・フロー"),
    "capital_expenditure": ("PurchaseOfPropertyPlantAndEquipment", "有形固定資産の取得"),
    "eps": ("BasicEarningsLossPerShare", "BasicEarningsPerShare", "１株当たり当期純利益"),
}


class EdinetClient:
    def __init__(self, api_key=None, cache_dir=None, timeout=30):
        self.api_key = api_key or os.getenv("EDINET_API_KEY")
        self.timeout = timeout
        self.cache_dir = Path(cache_dir or os.getenv("EDINET_CACHE_DIR", ".cache/edinet"))

    @property
    def configured(self):
        return bool(self.api_key)

    def _get(self, path, params):
        import requests

        if not self.configured:
            raise RuntimeError("EDINET_API_KEY is not configured")
        query = dict(params)
        query["Subscription-Key"] = self.api_key
        response = requests.get(f"{BASE_URL}/{path}", params=query, timeout=self.timeout)
        response.raise_for_status()
        return response

    def list_documents(self, target_date):
        day = str(target_date)
        cache = self.cache_dir / "lists" / f"{day}.json"
        if cache.exists():
            return json.loads(cache.read_text(encoding="utf-8"))
        payload = self._get("documents.json", {"date": day, "type": 2}).json()
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        return payload

    def find_company_documents(self, stock_code, lookback_days=None):
        code = str(stock_code).strip()[:4]
        lookback_days = int(lookback_days or os.getenv("EDINET_LOOKBACK_DAYS", "180"))
        found = []
        for offset in range(lookback_days + 1):
            day = date.today() - timedelta(days=offset)
            payload = self.list_documents(day.isoformat())
            for item in payload.get("results", []):
                sec_code = str(item.get("secCode") or "")[:4]
                if sec_code == code and item.get("docTypeCode") in {"120", "130", "140", "160"}:
                    found.append(item)
            if len(found) >= 4:
                break
        return sorted(found, key=lambda x: x.get("submitDateTime", ""), reverse=True)

    def download_csv_package(self, document_id):
        cache = self.cache_dir / "documents" / f"{document_id}.zip"
        if not cache.exists():
            response = self._get(f"documents/{document_id}", {"type": 5})
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_bytes(response.content)
        return cache.read_bytes()


def _read_csv_package(content):
    frames = []
    with zipfile.ZipFile(io.BytesIO(content)) as archive:
        for name in archive.namelist():
            if not name.lower().endswith(".csv"):
                continue
            raw = archive.read(name)
            for encoding in ("utf-16", "utf-8-sig", "cp932"):
                try:
                    frames.append(pd.read_csv(io.BytesIO(raw), encoding=encoding, dtype=str))
                    break
                except (UnicodeError, pd.errors.ParserError):
                    continue
    return frames


def _find_column(frame, candidates):
    normalized = {str(column).replace(" ", "").lower(): column for column in frame.columns}
    for candidate in candidates:
        key = candidate.replace(" ", "").lower()
        if key in normalized:
            return normalized[key]
    return None


def extract_financial_metrics(csv_frames):
    """EDINET CSVの表記揺れを許容して主要財務数値を抽出する。"""
    records = []
    for frame in csv_frames:
        element_col = _find_column(frame, ("要素ID", "elementid", "項目名", "element"))
        label_col = _find_column(frame, ("項目名", "label", "要素ID"))
        value_col = _find_column(frame, ("値", "value", "数値"))
        context_col = _find_column(frame, ("コンテキストID", "contextid", "相対年度"))
        if not value_col or not (element_col or label_col):
            continue
        for _, row in frame.iterrows():
            value_text = str(row.get(value_col, "")).replace(",", "").strip()
            try:
                value = float(value_text)
            except ValueError:
                continue
            records.append({
                "element": str(row.get(element_col, "")) if element_col else "",
                "label": str(row.get(label_col, "")) if label_col else "",
                "context": str(row.get(context_col, "")) if context_col else "",
                "value": value,
            })

    metrics = {}
    for metric, aliases in METRIC_ALIASES.items():
        matches = [record for record in records if any(
            alias.lower() in f"{record['element']} {record['label']}".lower() for alias in aliases
        )]
        current = [x for x in matches if "current" in x["context"].lower() or "当期" in x["context"]]
        selected = (current or matches)
        if selected:
            metrics[metric] = selected[0]["value"]
    if "operating_cash_flow" in metrics and "capital_expenditure" in metrics:
        metrics["free_cash_flow"] = metrics["operating_cash_flow"] - abs(metrics["capital_expenditure"])
    return metrics


def score_fundamentals(metrics):
    """取得できた財務値だけで構成し、カバレッジを別表示する保守的スコア。"""
    checks = []
    if metrics.get("revenue") is not None:
        checks.append(metrics["revenue"] > 0)
    if metrics.get("operating_income") is not None:
        checks.append(metrics["operating_income"] > 0)
    if metrics.get("net_income") is not None:
        checks.append(metrics["net_income"] > 0)
    if metrics.get("operating_cash_flow") is not None:
        checks.append(metrics["operating_cash_flow"] > 0)
    if metrics.get("free_cash_flow") is not None:
        checks.append(metrics["free_cash_flow"] > 0)
    if metrics.get("equity") is not None and metrics.get("total_assets"):
        checks.append(metrics["equity"] / metrics["total_assets"] >= 0.30)
    available = len(metrics)
    return {
        "score": int(round(100 * sum(checks) / len(checks))) if checks else None,
        "data_coverage": float(available / len(METRIC_ALIASES)),
        "evaluated_checks": len(checks),
        "interpretation": "available_items_only",
    }


def get_fundamental_analysis(stock_code, client=None):
    client = client or EdinetClient()
    if not client.configured:
        return {
            "available": False,
            "source": "EDINET_API_v2",
            "reason": "EDINET_API_KEY_not_configured",
        }
    documents = client.find_company_documents(stock_code)
    if not documents:
        return {"available": False, "source": "EDINET_API_v2", "reason": "no_documents_found"}
    csv_documents = [item for item in documents if str(item.get("csvFlag")) == "1"][:4]
    if not csv_documents:
        return {"available": False, "source": "EDINET_API_v2", "reason": "csv_package_unavailable"}
    history = []
    for item in csv_documents:
        metrics = extract_financial_metrics(_read_csv_package(client.download_csv_package(item["docID"])))
        if metrics:
            history.append({
                "document_id": item.get("docID"),
                "published_at": item.get("submitDateTime"),
                "document_type": item.get("docDescription"),
                "metrics": metrics,
            })
    if not history:
        return {"available": False, "source": "EDINET_API_v2", "reason": "financial_metrics_unavailable"}
    latest = history[0]
    metrics = dict(latest["metrics"])
    if metrics.get("revenue"):
        metrics["operating_margin"] = metrics.get("operating_income", 0) / metrics["revenue"]
        metrics["net_margin"] = metrics.get("net_income", 0) / metrics["revenue"]
    if metrics.get("total_assets"):
        metrics["equity_ratio"] = metrics.get("equity", 0) / metrics["total_assets"]
    if len(history) > 1:
        previous = history[1]["metrics"]
        for name in ("revenue", "operating_income", "net_income", "eps"):
            if name in metrics and previous.get(name) not in (None, 0):
                metrics[f"{name}_growth"] = metrics[name] / abs(previous[name]) - 1
    return {
        "available": bool(metrics),
        "source": "EDINET_API_v2",
        "document_id": latest["document_id"],
        "document_type": latest["document_type"],
        "published_at": latest["published_at"],
        "fetched_at": datetime.now().astimezone().isoformat(),
        "point_in_time_ready": True,
        "metrics": metrics,
        "assessment": score_fundamentals(metrics),
        "history": history,
    }
