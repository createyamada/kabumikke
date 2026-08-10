# 東証プライムランキングAPI

JPXが公開する上場銘柄一覧からプライム内国普通株式を抽出し、全銘柄を高速スクリーニングした後、上位候補へ既存の高度分析を適用します。

## 設定

```text
DATABASE_URL=postgresql://user:password@host:5432/database
DATABASE_RETRY_SECONDS=60
PRIME_RANKING_DIR=data/prime_ranking
PRIME_RANKING_WORKERS=3
PRIME_PROGRESS_WRITE_INTERVAL_SECONDS=1
ANALYSIS_MODEL_CACHE_ENABLED=true
ANALYSIS_MODEL_CACHE_DIR=.cache/models
```

`DATABASE_URL`が設定されている場合、ランキング本体、処理ステータス、プライム銘柄一覧、当日分析キャッシュをDBへ保存します。初回接続時に`kabumikke_store`テーブルを自動作成します。DB未設定または一時的な接続障害の場合は、従来のCSV・JSON・キャッシュファイルを使用します。

`PRIME_RANKING_WORKERS`は候補銘柄の並列分析数です。無料データ取得サービスへの負荷を抑えるため1～4に制限され、既定値は3です。

進捗は画面の表示精度を維持したまま、同一工程では既定1秒間隔または1ポイント以上進んだ場合に保存します。PostgreSQL/SQLiteでは更新をUPSERT 1回で処理します。

ランキングでは一次スクリーニング用に全銘柄の株価・出来高をまとめて取得し、高度分析へ進む候補についても10年分の株価、日経平均、TOPIX、為替、米国市場、業種ETFを一括取得して各ワーカーで共有します。

分析結果は入力株価の最終取引日ごとに`ANALYSIS_MODEL_CACHE_DIR`へ保存されます。同じ銘柄・同じ最終取引日の再分析ではモデル学習を省略し、新しい取引日のデータが追加されると別のキャッシュが生成されます。

DB利用時は分析結果の確認を市場データ取得より先に行います。また、同日分の市場履歴・前処理済み特徴量・ランキング用市場行列も再利用します。分析バージョンまたは特徴量バージョンが変わった場合は別キーになるため、古い計算結果が新しい分析へ混入しません。

JPXの一覧は月によって`.xls`または`.xlsx`で提供されるため、`requirements.txt`には`.xls`用の`xlrd`と`.xlsx`用の`openpyxl`を含めています。依存関係変更後は再インストールしてください。

```powershell
pip install -r requirements.txt
```

EDINET財務分析もランキングへ反映する場合は、`EDINET_API_KEY`も設定します。未設定でも株価・モデルによるランキングは生成できます。

## 更新開始

```http
POST /api/prime-ranking/refresh?limit=10&shortlist_size=50
```

処理はバックグラウンドで開始され、すぐに`202 Accepted`を返します。`shortlist_size`は高度分析を行う候補数で、10～200件を指定できます。

## 状態確認

```http
GET /api/prime-ranking/status
```

`status`は`not_started`、`running`、`completed`、`failed`のいずれかです。

状態レスポンスの`refresh_allowed`が`false`の場合、フロントの更新ボタンを無効化してください。

```json
{
  "refresh_allowed": false,
  "refresh_block_reason": "ranking_already_generated_today",
  "latest_generated_date": "2026-08-05",
  "today_jst": "2026-08-05"
}
```

判定日はサーバーのローカル時刻ではなく、日本時間`Asia/Tokyo`で統一しています。当日分のCSVが存在する場合は、画面側だけでなく更新API側でも二重生成を拒否します。

## 最新ランキング

```http
GET /api/prime-ranking?limit=10
```

分析実行中は前回完成済みの`prime_ranking_latest.csv`を返します。新しいCSVは一時ファイルへ出力し、必須列・順位・銘柄重複を検証した後に`os.replace()`で置換します。処理失敗時は前回CSVを削除しません。

CSVには日付`generated_date`と日時`analyzed_at`の両方を保存します。

## 出力ファイル

```text
data/prime_ranking/
├── prime_ranking_latest.csv
├── prime_universe_latest.csv
├── analysis_status.json
└── work/
```

ランキングCSVは最新の候補一覧だけを保持します。JPX一覧も最新ファイルへ置き換えられます。EDINETのキャッシュは再取得負荷を避けるため別途保持します。

毎朝の自動更新は、OSのタスクスケジューラやcronから更新APIを呼び出してください。
