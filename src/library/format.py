import pandas as pd
from datetime import datetime
from datetime import timedelta
from library import config


def merge_all_company_info(infos: list):
    """
    リストの要素数分データフレームを紐づける

    Parameters:
    - infos: list 企業情報
    Returns:
    - result: DataFrame 企業情報
    """

    for index, info in enumerate(infos):
        if index == 0:
            merged_df = dataframe_index_to_clumn(info)
        else:
            if info.index.name != 'Date':
                info = dataframe_index_to_clumn(info)

            merged_df = pd.merge(merged_df, info, on='Date', how="left")

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

    data_technical = merged_df[config.EXPLANATORY_VARIABLES].dropna(how='any')

    return data_technical


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
    # タイムゾーン情報を削除する
    result['Date'] = result['Date'].dt.tz_localize(None)
    return result


def get_divided_data(data):
    """
    学習用、検証用にデータを分ける

    Parameters:
    - data データフレーム
    Returns:
    - result {学習用,検証用}
    """
    dates = get_divided_date(data.index.tolist(),365)


    # それぞれデータを作成
    train = data[dates['start'] : dates['start_end']]
    test = data[dates['end_start'] :]
    # 学習用データとテストデータそれぞれを説明変数と目的変数に分離する
    X_train = train.drop(columns=['Close_next'])
    Y_train = train['Close_next']
    X_test = test.drop(columns=['Close_next'])
    Y_test = pd.DataFrame(test['Close_next'],columns=['Close_next'])

    return {
        'X_train':X_train,
        'Y_train':Y_train,
        'X_test':X_test,
        'Y_test':Y_test,
    }


def get_divided_date(data,days):
    """
    データフレームの年月日の指定日付前の年月日を取得

    Parameters:
    - data 配列
    - days 何日前か
    Returns:
    - result {最初の日、指定日の前日、指定日、最後の日}
    """
    return {
        'start':data[0].strftime('%Y-%m-%d'),
        'start_end': (data[-1] - timedelta(days=days) - timedelta(days=1)).strftime('%Y-%m-%d'),
        'end_start': (data[-1] - timedelta(days=days)).strftime('%Y-%m-%d'),
        'end':data[-1].strftime('%Y-%m-%d')
    }





