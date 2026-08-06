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
    'sma5_sma25_gap',
    'sma25_sma70_gap',
    'sma5_slope5',
    'sma25_slope5',
    'sma70_slope5',
    'golden_cross',
    'dead_cross',
    'days_since_golden_cross',
    'days_since_dead_cross',
    'ma_order_score',
    'perfect_order_bull',
    'perfect_order_bear',
    'distance_from_high20',
    'distance_from_high60',
    'distance_from_high252',
    'distance_from_low20',
    'distance_from_low60',
    'higher_high',
    'higher_low',
    'resistance_gap20',
    'support_gap20',
    'breakout_up20',
    'breakout_down20',
    'range_width20',
    'range_width60',
    'candle_body_ratio',
    'upper_shadow_ratio',
    'lower_shadow_ratio',
    'gap_rate',
    'consecutive_bullish',
    'consecutive_bearish',
    'bullish_engulfing',
    'bearish_engulfing',
    'doji',
    'stochastic_k',
    'stochastic_d',
    'plus_di14',
    'minus_di14',
    'adx14',
    'roc10',
    'cci20',
    'obv_slope20',
    'mfi14',
    'vwap20_gap',
    'ichimoku_tenkan_gap',
    'ichimoku_kijun_gap',
    'ichimoku_tenkan_kijun_gap',
    'ichimoku_cloud_position',
    'ichimoku_cloud_width',
    'volume_profile_poc_gap',
    'volume_profile_value_area_width',
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
