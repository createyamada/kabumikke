# EDINET連携

EDINET API Version 2で発行したAPIキーを、リポジトリへコミットせず環境変数に設定します。

```text
EDINET_API_KEY=発行されたAPIキー
EDINET_CACHE_DIR=.cache/edinet
EDINET_LOOKBACK_DAYS=180
```

APIキーが未設定、書類が未取得、または一時的にEDINETへ接続できない場合も、株価予測APIは停止しません。`fundamental_analysis.available` が `false` になり、`reason` に理由が入ります。

取得済みの日別書類一覧とCSV ZIPは `EDINET_CACHE_DIR` に保存されます。初回は対象期間の日別一覧取得に時間がかかるため、常時運用では夜間処理でキャッシュを更新してください。

株価予測レスポンスに以下が追加されます。

- `fundamental_analysis.metrics`: 売上高、利益、資産、純資産、CF、EPS、利益率、成長率
- `fundamental_analysis.assessment`: 取得可能項目だけを使った財務スコアとデータカバレッジ
- `fundamental_analysis.history`: 公表日時付きの取得履歴

財務分析だけを取得する場合は次のAPIを利用できます。

```text
GET /api/stock_price_prediction/fundamentals?code=5802
```

横断バックテストは次のAPIで実行します。

```text
GET /api/stock_price_prediction/cross-sectional-backtest?codes=5802,6501,6702&period=5y&top_n=2
```

最大200銘柄まで指定できます。シグナル算出日の翌営業日から保有するため、同日終値の未来情報は使用しません。
