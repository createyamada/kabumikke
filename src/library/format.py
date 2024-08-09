import pandas as pd
from datetime import datetime
from datetime import timedelta


# 説明変数として取得するデータ名の配列
REQUIRE_DATA = [
    'High',
    'Low',
    'Open',
    'Close',
    'Body',
    'Close_diff',
    'SMA5',
    'SMA25',
    'SMA70',
    'Close_next'
]

def merge_all_company_info(infos:list):
    """
    リストの要素数分データフレームを紐づける

    Parameters:
    - info:list 企業情報
    Returns:
    - result:Dataframe 企業情報
    """


    # リストの要素数分データフレームを紐づける
    for index, info in enumerate(infos):
        if index == 0:
            # 先頭データの場合
            merged_df = dataframe_index_to_clumn(info)
            continue

        else:
            # 先頭データでない場合
            if info.index.name != 'Date':
                # インデックスにDateが設定されていない場合
                info = dataframe_index_to_clumn(info.T)
            else:
                info = dataframe_index_to_clumn(info)


        # 前回のデータフレームと日付をキーにしてマージ
        merged_df = pd.merge(merged_df,info,on='Date',how="left")

    # DateをDatetime64型に変換
    merged_df['Date'] = pd.to_datetime(merged_df['Date'])
    # 曜日情報を追加(月曜:0, 火曜:1, 水曜:2, 木曜:3, 金曜:4, 土曜:5, 日曜:6)
    merged_df['weekday'] = merged_df['Date'].dt.weekday
    # 先頭の日付を取得
    start = merged_df.iloc[0]['Date']
    # 先頭の月曜日を基準に週を追加
    merged_df['weeks'] = (merged_df['Date'] - start) // timedelta(weeks=1)
    # 日付をインデックスにセット
    merged_df.set_index(keys='Date', inplace=True)
    # データの並び替え
    merged_df.sort_values(by='Date', ascending=True, inplace=True)
    # 始値と終値の差分を追加
    merged_df['Body'] = merged_df['Open'] - merged_df['Close']
    # 翌日の終値と本日の終値の差分を追加する
    merged_df['Close_diff'] = merged_df['Close'].diff(1)
    # 目的変数となる翌日の終値を追加
    merged_df['Close_next'] = merged_df['Close'].shift(-1)
    # 移動平均を追加
    merged_df['SMA5'] = merged_df['Close'].rolling(5).mean()
    merged_df['SMA25'] = merged_df['Close'].rolling(25).mean()
    merged_df['SMA70'] = merged_df['Close'].rolling(70).mean()

    # 説明変数として取得するデータだけを抽出
    data_technical = merged_df[REQUIRE_DATA]

    # TODO 要検討
    # 欠損値削除
    data_technical = data_technical.dropna(how='any')

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
    # print('data')
    # print(data.index.tolist())
    dates = get_divided_date(data.index.tolist(),365)
    # それぞれデータを作成
    train = data[dates['start'] : dates['start_end']]
    test = data[dates['end_start'] :]
    # それぞれを説明変数と目的変数に分離する
    X_train = train.drop(columns=['Close_next'])
    Y_train = train['Close_next']

    X_test = train.drop(columns=['Close_next'])
    Y_test = train['Close_next']


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





