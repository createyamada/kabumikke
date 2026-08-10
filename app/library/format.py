import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from library import config


def _days_since_signal(signal):
    """当日を0として直近シグナルからの営業日数を返す。未発生は-1。"""
    result = []
    elapsed = -1
    for active in signal.fillna(False).astype(bool):
        if active:
            elapsed = 0
        elif elapsed >= 0:
            elapsed += 1
        result.append(elapsed)
    return pd.Series(result, index=signal.index, dtype=float)


def _consecutive_count(condition):
    count = 0
    values = []
    for active in condition.fillna(False).astype(bool):
        count = count + 1 if active else 0
        values.append(count)
    return pd.Series(values, index=condition.index, dtype=float)


def _volume_profile_features(close, volume, window=60, bins=12):
    """過去window日だけを使用し、出来高集中価格と70%価値領域幅を近似する。"""
    poc_gap = np.full(len(close), np.nan)
    value_area_width = np.full(len(close), np.nan)
    close_values = close.to_numpy(dtype=float)
    volume_values = volume.fillna(0).to_numpy(dtype=float)
    for end in range(window - 1, len(close)):
        prices = close_values[end - window + 1:end + 1]
        weights = volume_values[end - window + 1:end + 1]
        valid = np.isfinite(prices) & np.isfinite(weights) & (weights >= 0)
        prices, weights = prices[valid], weights[valid]
        if len(prices) < window // 2 or prices.max() <= prices.min() or weights.sum() <= 0:
            continue
        edges = np.linspace(prices.min(), prices.max(), bins + 1)
        bucket = np.clip(np.digitize(prices, edges[1:-1]), 0, bins - 1)
        profile = np.bincount(bucket, weights=weights, minlength=bins)
        centers = (edges[:-1] + edges[1:]) / 2
        poc = centers[int(np.argmax(profile))]
        order = np.argsort(profile)[::-1]
        selected = []
        accumulated = 0.0
        for index in order:
            selected.append(index)
            accumulated += profile[index]
            if accumulated >= profile.sum() * 0.70:
                break
        low, high = edges[min(selected)], edges[max(selected) + 1]
        poc_gap[end] = close_values[end] / poc - 1 if poc else np.nan
        value_area_width[end] = (high - low) / close_values[end] if close_values[end] else np.nan
    return (
        pd.Series(poc_gap, index=close.index),
        pd.Series(value_area_width, index=close.index),
    )

def merge_all_company_info(infos: list):
    """
    リストの要素数分データフレームを結合する

    Parameters:
    - infos: list 企業情報
    Returns:
    - result: DataFrame 企業情報
    """
    merged_df = None

    for index, info in enumerate(infos):
        # Callers may share benchmark frames across ranking workers.
        # Keep feature construction side-effect free without deep-copying all inputs.
        info = info.copy(deep=False)
        if info.empty:
            print(f"Warning: DataFrame at index {index} is empty. Skipping...")
            continue

        # Date カラムがない場合、インデックスをリセットして確保
        if 'Date' not in info.columns:
            info = dataframe_index_to_clumn(info)

        # Date カラムの型を統一
        info['Date'] = pd.to_datetime(info['Date']).dt.tz_localize(None)

        if merged_df is None:
            merged_df = info
        else:
            merged_df = pd.merge(merged_df, info, on='Date', how="left")

    if merged_df is None or merged_df.empty:
        raise ValueError("Error: No valid dataframes found to merge.")

    merged_df['Date'] = pd.to_datetime(merged_df['Date'])
    merged_df['weekday'] = merged_df['Date'].dt.weekday

    start = merged_df.iloc[0]['Date']
    merged_df['weeks'] = (merged_df['Date'] - start) // timedelta(weeks=1)

    merged_df.set_index(keys='Date', inplace=True)
    merged_df.sort_values(by='Date', ascending=True, inplace=True)

    # 日本株の引け時点では同じ日付の米国終値と日次為替終値は未確定になり得る。
    # 日付結合後に1行遅らせ、予測時点で確実に既知だった値だけを利用する。
    # 米国先物の同日終値も確定時刻が曖昧なので、日次モデルには混ぜない。
    for columns in (('dow_open', 'dow_close'), ('jpy_open', 'jpy_close')):
        available = [column for column in columns if column in merged_df.columns]
        if available:
            merged_df[available] = merged_df[available].shift(1).ffill()

    merged_df['Body'] = (merged_df['Open'] - merged_df['Close']).fillna(0)
    merged_df['Close_diff'] = merged_df['Close'].diff(1).fillna(0)
    merged_df['Close_next'] = merged_df['Close'].shift(-1)

    merged_df['SMA5'] = merged_df['Close'].rolling(5, min_periods=1).mean()
    merged_df['SMA25'] = merged_df['Close'].rolling(25, min_periods=1).mean()
    merged_df['SMA70'] = merged_df['Close'].rolling(70, min_periods=1).mean()

    # 価格水準ではなく、銘柄や相場局面をまたいで比較できる比率を特徴量にする。
    merged_df['return_1d'] = merged_df['Close'].pct_change()
    merged_df['return_5d'] = merged_df['Close'].pct_change(5)
    merged_df['return_20d'] = merged_df['Close'].pct_change(20)
    merged_df['intraday_return'] = merged_df['Close'] / merged_df['Open'] - 1
    merged_df['sma5_gap'] = merged_df['Close'] / merged_df['SMA5'] - 1
    merged_df['sma25_gap'] = merged_df['Close'] / merged_df['SMA25'] - 1
    merged_df['sma70_gap'] = merged_df['Close'] / merged_df['SMA70'] - 1

    close_delta = merged_df['Close'].diff()
    average_gain = close_delta.clip(lower=0).rolling(14).mean()
    average_loss = -close_delta.clip(upper=0).rolling(14).mean()
    relative_strength = average_gain / average_loss.replace(0, np.nan)
    merged_df['rsi14'] = (100 - 100 / (1 + relative_strength)).fillna(50) / 100

    ema12 = merged_df['Close'].ewm(span=12, adjust=False).mean()
    ema26 = merged_df['Close'].ewm(span=26, adjust=False).mean()
    merged_df['macd'] = (ema12 - ema26) / merged_df['Close']

    previous_close = merged_df['Close'].shift(1)
    true_range = pd.concat([
        merged_df['High'] - merged_df['Low'],
        (merged_df['High'] - previous_close).abs(),
        (merged_df['Low'] - previous_close).abs(),
    ], axis=1).max(axis=1)
    merged_df['atr14_rate'] = true_range.rolling(14).mean() / merged_df['Close']

    rolling_std = merged_df['Close'].rolling(20).std()
    merged_df['bollinger_position'] = (
        (merged_df['Close'] - merged_df['SMA25']) / (2 * rolling_std.replace(0, np.nan))
    )
    merged_df['volatility20'] = merged_df['return_1d'].rolling(20).std()
    merged_df['volume_change'] = merged_df['Volume'].pct_change()
    merged_df['volume_ratio20'] = merged_df['Volume'] / merged_df['Volume'].rolling(20).mean() - 1

    # 移動平均線の相互関係、クロス、傾き、並び順。
    merged_df['sma5_sma25_gap'] = merged_df['SMA5'] / merged_df['SMA25'] - 1
    merged_df['sma25_sma70_gap'] = merged_df['SMA25'] / merged_df['SMA70'] - 1
    merged_df['sma5_slope5'] = merged_df['SMA5'].pct_change(5)
    merged_df['sma25_slope5'] = merged_df['SMA25'].pct_change(5)
    merged_df['sma70_slope5'] = merged_df['SMA70'].pct_change(5)
    above = merged_df['SMA5'] > merged_df['SMA25']
    previous_above = above.shift(1, fill_value=False).astype(bool)
    merged_df['golden_cross'] = (above & ~previous_above).astype(float)
    merged_df['dead_cross'] = (~above & previous_above).astype(float)
    merged_df['days_since_golden_cross'] = _days_since_signal(merged_df['golden_cross'] > 0)
    merged_df['days_since_dead_cross'] = _days_since_signal(merged_df['dead_cross'] > 0)
    bullish_order = (merged_df['SMA5'] > merged_df['SMA25']) & (merged_df['SMA25'] > merged_df['SMA70'])
    bearish_order = (merged_df['SMA5'] < merged_df['SMA25']) & (merged_df['SMA25'] < merged_df['SMA70'])
    merged_df['ma_order_score'] = np.select([bullish_order, bearish_order], [1.0, -1.0], default=0.0)
    merged_df['perfect_order_bull'] = (bullish_order & (merged_df['sma5_slope5'] > 0) & (merged_df['sma25_slope5'] > 0)).astype(float)
    merged_df['perfect_order_bear'] = (bearish_order & (merged_df['sma5_slope5'] < 0) & (merged_df['sma25_slope5'] < 0)).astype(float)

    # トレンド・価格帯。ブレイクアウト判定は当日を除く過去水準と比較する。
    high20, high60, high252 = (merged_df['High'].rolling(window).max() for window in (20, 60, 252))
    low20, low60 = (merged_df['Low'].rolling(window).min() for window in (20, 60))
    merged_df['distance_from_high20'] = merged_df['Close'] / high20 - 1
    merged_df['distance_from_high60'] = merged_df['Close'] / high60 - 1
    merged_df['distance_from_high252'] = merged_df['Close'] / high252 - 1
    merged_df['distance_from_low20'] = merged_df['Close'] / low20 - 1
    merged_df['distance_from_low60'] = merged_df['Close'] / low60 - 1
    merged_df['higher_high'] = (merged_df['High'].rolling(5).max() > merged_df['High'].rolling(5).max().shift(5)).astype(float)
    merged_df['higher_low'] = (merged_df['Low'].rolling(5).min() > merged_df['Low'].rolling(5).min().shift(5)).astype(float)
    prior_resistance = merged_df['High'].rolling(20).max().shift(1)
    prior_support = merged_df['Low'].rolling(20).min().shift(1)
    merged_df['resistance_gap20'] = merged_df['Close'] / prior_resistance - 1
    merged_df['support_gap20'] = merged_df['Close'] / prior_support - 1
    merged_df['breakout_up20'] = (merged_df['Close'] > prior_resistance).astype(float)
    merged_df['breakout_down20'] = (merged_df['Close'] < prior_support).astype(float)
    merged_df['range_width20'] = (high20 - low20) / merged_df['Close']
    merged_df['range_width60'] = (high60 - low60) / merged_df['Close']

    # ローソク足形状と連続性。
    candle_range = (merged_df['High'] - merged_df['Low']).replace(0, np.nan)
    candle_top = merged_df[['Open', 'Close']].max(axis=1)
    candle_bottom = merged_df[['Open', 'Close']].min(axis=1)
    merged_df['candle_body_ratio'] = ((merged_df['Close'] - merged_df['Open']).abs() / candle_range).fillna(0)
    merged_df['upper_shadow_ratio'] = ((merged_df['High'] - candle_top) / candle_range).fillna(0)
    merged_df['lower_shadow_ratio'] = ((candle_bottom - merged_df['Low']) / candle_range).fillna(0)
    merged_df['gap_rate'] = merged_df['Open'] / merged_df['Close'].shift(1) - 1
    bullish_candle = merged_df['Close'] > merged_df['Open']
    bearish_candle = merged_df['Close'] < merged_df['Open']
    merged_df['consecutive_bullish'] = _consecutive_count(bullish_candle)
    merged_df['consecutive_bearish'] = _consecutive_count(bearish_candle)
    previous_top = merged_df[['Open', 'Close']].max(axis=1).shift(1)
    previous_bottom = merged_df[['Open', 'Close']].min(axis=1).shift(1)
    merged_df['bullish_engulfing'] = (bullish_candle & bearish_candle.shift(1).fillna(False) & (candle_top >= previous_top) & (candle_bottom <= previous_bottom)).astype(float)
    merged_df['bearish_engulfing'] = (bearish_candle & bullish_candle.shift(1).fillna(False) & (candle_top >= previous_top) & (candle_bottom <= previous_bottom)).astype(float)
    merged_df['doji'] = (merged_df['candle_body_ratio'] <= 0.10).astype(float)

    # ストキャスティクス、DMI/ADX、ROC、CCI。
    lowest14 = merged_df['Low'].rolling(14).min()
    highest14 = merged_df['High'].rolling(14).max()
    merged_df['stochastic_k'] = ((merged_df['Close'] - lowest14) / (highest14 - lowest14).replace(0, np.nan)).fillna(0.5)
    merged_df['stochastic_d'] = merged_df['stochastic_k'].rolling(3).mean()
    up_move = merged_df['High'].diff()
    down_move = -merged_df['Low'].diff()
    plus_dm = pd.Series(np.where((up_move > down_move) & (up_move > 0), up_move, 0.0), index=merged_df.index)
    minus_dm = pd.Series(np.where((down_move > up_move) & (down_move > 0), down_move, 0.0), index=merged_df.index)
    atr14 = true_range.rolling(14).mean().replace(0, np.nan)
    merged_df['plus_di14'] = plus_dm.rolling(14).mean() / atr14
    merged_df['minus_di14'] = minus_dm.rolling(14).mean() / atr14
    dx = ((merged_df['plus_di14'] - merged_df['minus_di14']).abs() / (merged_df['plus_di14'] + merged_df['minus_di14']).replace(0, np.nan)).fillna(0)
    merged_df['adx14'] = dx.rolling(14).mean().fillna(0)
    merged_df['roc10'] = merged_df['Close'].pct_change(10)
    typical_price = (merged_df['High'] + merged_df['Low'] + merged_df['Close']) / 3
    typical_mean = typical_price.rolling(20).mean()
    mean_deviation = typical_price.rolling(20).apply(lambda values: np.mean(np.abs(values - np.mean(values))), raw=True)
    merged_df['cci20'] = ((typical_price - typical_mean) / (0.015 * mean_deviation.replace(0, np.nan))).fillna(0)

    # OBV、MFI、VWAP。
    direction = np.sign(merged_df['Close'].diff()).fillna(0)
    obv = (direction * merged_df['Volume'].fillna(0)).cumsum()
    merged_df['obv_slope20'] = (obv.diff(20) / (merged_df['Volume'].rolling(20).mean() * 20).replace(0, np.nan)).fillna(0)
    raw_money_flow = typical_price * merged_df['Volume']
    positive_flow = raw_money_flow.where(typical_price.diff() > 0, 0.0).rolling(14).sum()
    negative_flow = raw_money_flow.where(typical_price.diff() < 0, 0.0).rolling(14).sum()
    money_ratio = positive_flow / negative_flow.replace(0, np.nan)
    mfi = 1 - 1 / (1 + money_ratio)
    mfi = mfi.mask((negative_flow == 0) & (positive_flow > 0), 1.0)
    mfi = mfi.mask((positive_flow == 0) & (negative_flow > 0), 0.0)
    merged_df['mfi14'] = mfi.fillna(0.5)
    rolling_vwap20 = raw_money_flow.rolling(20).sum() / merged_df['Volume'].rolling(20).sum().replace(0, np.nan)
    merged_df['vwap20_gap'] = merged_df['Close'] / rolling_vwap20 - 1

    # 一目均衡表。先行スパンは26日前に計算された値だけを現在位置へ利用する。
    tenkan = (merged_df['High'].rolling(9).max() + merged_df['Low'].rolling(9).min()) / 2
    kijun = (merged_df['High'].rolling(26).max() + merged_df['Low'].rolling(26).min()) / 2
    senkou_a = ((tenkan + kijun) / 2).shift(26)
    senkou_b = ((merged_df['High'].rolling(52).max() + merged_df['Low'].rolling(52).min()) / 2).shift(26)
    cloud_top = pd.concat([senkou_a, senkou_b], axis=1).max(axis=1)
    cloud_bottom = pd.concat([senkou_a, senkou_b], axis=1).min(axis=1)
    merged_df['ichimoku_tenkan_gap'] = merged_df['Close'] / tenkan - 1
    merged_df['ichimoku_kijun_gap'] = merged_df['Close'] / kijun - 1
    merged_df['ichimoku_tenkan_kijun_gap'] = tenkan / kijun - 1
    merged_df['ichimoku_cloud_position'] = np.select(
        [merged_df['Close'] > cloud_top, merged_df['Close'] < cloud_bottom], [1.0, -1.0], default=0.0
    )
    merged_df['ichimoku_cloud_width'] = (cloud_top - cloud_bottom) / merged_df['Close']

    merged_df['volume_profile_poc_gap'], merged_df['volume_profile_value_area_width'] = _volume_profile_features(
        merged_df['Close'], merged_df['Volume'], window=60
    )
    merged_df['nikkei_return'] = merged_df['nikkei_close'].pct_change()
    # TOPIXが取得できない日は日経平均の変化率を市場代理値として利用する。
    if 'topix_close' in merged_df.columns:
        merged_df['topix_return'] = merged_df['topix_close'].pct_change()
    else:
        merged_df['topix_return'] = merged_df['nikkei_return']
    merged_df['dow_return'] = merged_df['dow_close'].pct_change()
    merged_df['jpy_return'] = merged_df['jpy_close'].pct_change()
    if 'sector_close' in merged_df.columns:
        merged_df['sector_return'] = merged_df['sector_close'].pct_change()
        merged_df['sector_relative_strength_20d'] = (
            (1 + merged_df['return_1d']).rolling(20).apply(np.prod, raw=True)
            / (1 + merged_df['sector_return']).rolling(20).apply(np.prod, raw=True) - 1
        )
        merged_df['sector_benchmark_source'] = 'topix_17_etf'
    else:
        merged_df['sector_return'] = merged_df['topix_return']
        merged_df['sector_relative_strength_20d'] = merged_df['return_20d'] - (
            (1 + merged_df['topix_return']).rolling(20).apply(np.prod, raw=True) - 1
        )
        merged_df['sector_benchmark_source'] = 'topix_fallback'
    # 配列に含まれる列名のみを抽出
    # merged_df = merged_df[config.EXPLANATORY_VARIABLES]
    return merged_df

def dataframe_index_to_clumn(data):
    """
    データフレームの日付インデックスをカラムにする

    Parameters:
    - data データフレーム
    Returns:
    - result 日付カラムを加えたデータフレーム
    """
    result = data.reset_index().rename(columns={'index': 'Date'})
    # タイムゾーン情報を削除し、型を統一
    result['Date'] = pd.to_datetime(result['Date']).dt.tz_localize(None)
    return result

def get_divided_data(data):
    """
    学習用、検証用にデータを分ける

    Parameters:
    - data データフレーム
    Returns:
    - result {学習用,検証用}
    """
    if data.empty:
        raise ValueError("分析対象の株価データがありません。")

    data = data.copy().sort_index()
    features = config.EXPLANATORY_VARIABLES_ANALYSIS
    required_columns = set(features + ['Close', 'Close_next'])
    missing_columns = sorted(required_columns.difference(data.columns))
    if missing_columns:
        raise ValueError(f"分析に必要な列がありません: {', '.join(missing_columns)}")

    feature_data = data[features].replace([np.inf, -np.inf], np.nan).ffill()
    last_data = feature_data.iloc[-1]
    if last_data.isna().any():
        missing = ', '.join(last_data.index[last_data.isna()].tolist())
        raise ValueError(f"最新日の説明変数を補完できません: {missing}")

    target_return = data['Close_next'] / data['Close'] - 1
    labeled = pd.concat([
        feature_data,
        target_return.rename('target_return'),
        data['Close_next'],
    ], axis=1).dropna(how="any")
    if len(labeled) < 50:
        raise ValueError("分析に必要な履歴が不足しています（50営業日以上必要です）。")

    # 最長約1年、かつ全体の20%をホールドアウトして時系列順に評価する。
    test_size = min(252, max(20, len(labeled) // 5))
    if len(labeled) - test_size < 30:
        test_size = len(labeled) - 30
    train = labeled.iloc[:-test_size]
    test = labeled.iloc[-test_size:]

    X_train = train[features]
    Y_train = train['target_return']
    X_test = test[features]
    Y_test = test['target_return']

    return {
        'X_train': X_train,
        'Y_train': Y_train,
        'X_test': X_test,
        'Y_test': Y_test,
        'last_data': last_data,
        'X_all': labeled[features],
        'Y_all': labeled['target_return'],
        'actual_close_test': test['Close_next'],
        'current_close_test': data.loc[test.index, 'Close'],
        'last_close': float(data.iloc[-1]['Close']),
        # 複数予測期間の教師データを作るための内部データ。APIには直接返さない。
        'source_data': data,
        'feature_data': feature_data,
    }

def get_divided_date(data, days):
    """
    データフレームの年月日の指定日付前の年月日を取得

    Parameters:
    - data 配列
    - days 何日前か
    Returns:
    - result {最初の日、指定日の前日、指定日、最後の日}
    """
    data.sort()
    return {
        'start': data[0].strftime('%Y-%m-%d'),
        'start_end': (data[-1] - timedelta(days=days) - timedelta(days=1)).strftime('%Y-%m-%d'),
        'end_start': (data[-1] - timedelta(days=days)).strftime('%Y-%m-%d'),
        'end': data[-1].strftime('%Y-%m-%d')
    }
