# 東証プライムランキングAPI

JPXが公開する上場銘柄一覧からプライム内国普通株式を抽出し、全銘柄を高速スクリーニングした後、上位候補へ既存の高度分析を適用します。

## 設定

```text
PRIME_RANKING_DIR=data/prime_ranking
```

EDINET財務分析もランキングへ反映する場合は、`EDINET_API_KEY`も設定します。未設定でも株価・モデル・TDAによるランキングは生成できます。

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

## 最新ランキング

```http
GET /api/prime-ranking?limit=10
```

分析実行中は前回完成済みの`prime_ranking_latest.csv`を返します。新しいCSVは一時ファイルへ出力し、必須列・順位・銘柄重複を検証した後に`os.replace()`で置換します。処理失敗時は前回CSVを削除しません。

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
