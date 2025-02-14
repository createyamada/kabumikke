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
    'nikkei_open',
    'nikkei_close',
    'dow_open',
    'dow_close',
    'jpy_open',
    'jpy_close',
]

# 目的変数として取得するデータ名の配列
OBJECT_VARIABLES = [
    'Close_next'
]
