import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from library import config

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

    # 米国市場の当日値が未確定なら先物で補い、それもなければ直近値を使用する。
    if pd.isna(merged_df.iloc[-1]['dow_open']):
        if pd.notna(merged_df.iloc[-1].get('mini_dow_open')):
            merged_df.loc[merged_df.index[-1], 'dow_open'] = merged_df.iloc[-1]['mini_dow_open']
            merged_df.loc[merged_df.index[-1], 'dow_close'] = merged_df.iloc[-1]['mini_dow_close']
    merged_df[['dow_open', 'dow_close']] = merged_df[['dow_open', 'dow_close']].ffill()

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
    merged_df['nikkei_return'] = merged_df['nikkei_close'].pct_change()
    merged_df['dow_return'] = merged_df['dow_close'].pct_change()
    merged_df['jpy_return'] = merged_df['jpy_close'].pct_change()
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
    data.reset_index(inplace=True)
    result = data.rename(columns={'index': 'Date'})
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
