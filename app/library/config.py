# 説明変数として取得するデータ名の配列
EXPLANATORY_VARIABLES = [
    'High',
    'Low',
    'Open',
    'Close',
    'Body',
    'Close_diff',
    'SMA5',
    'SMA25',
    'SMA70',
    'Volume',
    'Close_next',
    'nikkei_open',
    'nikkei_close',
    'dow_open',
    'dow_close',
    'jpy_open',
    'jpy_close',
]

# 予測に使用する説明変数
EXPLANATORY_VARIABLES_ANALYSIS = [
    'return_1d',
    'return_5d',
    'return_20d',
    'intraday_return',
    'sma5_gap',
    'sma25_gap',
    'sma70_gap',
    'rsi14',
    'macd',
    'atr14_rate',
    'bollinger_position',
    'volatility20',
    'volume_change',
    'volume_ratio20',
    'nikkei_return',
    'topix_return',
    'sector_return',
    'sector_relative_strength_20d',
    'dow_return',
    'jpy_return',
]

# 目的変数として取得するデータ名の配列
OBJECT_VARIABLES = [
    'target_return'
]

# バックテストで片道売買時に控除する概算コスト（10bp）。
TRANSACTION_COST_RATE = 0.001

# 終値確定後の発注が翌営業日に約定する際の概算スリッページ（5bp）。
SLIPPAGE_RATE = 0.0005

# 適応的予測区間の設定。
PREDICTION_INTERVAL_COVERAGE = 0.80
ADAPTIVE_CONFORMAL_LEARNING_RATE = 0.02
