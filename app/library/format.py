import pandas as pd
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

    merged_df['Body'] = (merged_df['Open'] - merged_df['Close']).fillna(0)
    merged_df['Close_diff'] = merged_df['Close'].diff(1).fillna(0)
    merged_df['Close_next'] = merged_df['Close'].shift(-1)

    merged_df['SMA5'] = merged_df['Close'].rolling(5, min_periods=1).mean()
    merged_df['SMA25'] = merged_df['Close'].rolling(25, min_periods=1).mean()
    merged_df['SMA70'] = merged_df['Close'].rolling(70, min_periods=1).mean()
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
    dates = get_divided_date(data.index.tolist(), 365)

    if pd.isna(data[config.EXPLANATORY_VARIABLES_ANALYSIS].iloc[-1]["dow_open"]):
        data.loc[data.index[-1], 'dow_open'] = data.iloc[-1]["mini_dow_open"]
        data.loc[data.index[-1], 'dow_close'] = data.iloc[-1]["mini_dow_close"]

    last_data = data[config.EXPLANATORY_VARIABLES_ANALYSIS].iloc[-1].values.reshape(1, -1)

    # それぞれデータを作成
    train = data[config.EXPLANATORY_VARIABLES][dates['start']: dates['start_end']].dropna(how="any")
    test = data[config.EXPLANATORY_VARIABLES][dates['end_start']:].dropna(how="any")


    # 学習用データとテストデータそれぞれを説明変数と目的変数に分離する
    X_train = train.drop(columns=['Close_next'])
    Y_train = train['Close_next']
    X_test = test.drop(columns=['Close_next'])
    Y_test = pd.DataFrame(test['Close_next'], columns=['Close_next'])

    return {
        'X_train': X_train,
        'Y_train': Y_train,
        'X_test': X_test,
        'Y_test': Y_test,
        'last_data': data[config.EXPLANATORY_VARIABLES_ANALYSIS].iloc[-1].values.reshape(1, -1),
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
