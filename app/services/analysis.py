from fastapi import FastAPI, HTTPException
# 線形回帰モデルのLinearRegressionをインポート
from sklearn.linear_model import LinearRegression
# 時系列分割のためTimeSeriesSplitのインポート
from sklearn.model_selection import TimeSeriesSplit
# 予測精度検証のためMSEをインポート
from sklearn.metrics import mean_squared_error as mse
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pandas as pd
from datetime import datetime
from datetime import timedelta
from library import format
from library import config
import yfinance as yf


def get_analysis_data(company):
    """
    分析のためのデータを取得する

    Parameters:
    - company:str 特定の企業データ(yfinanceで取得)

    Returns:
    - result: 全分析データフレームの配列
    """
    result = [];


    # 企業の株価時系列
    result.append(company.history(period="max"))


    # 日経平均株価を取得する
    nikkei = yf.Ticker("^N225")
    nikkei_info = nikkei.history(period="max",prepost=True,actions=False)
    nikkei_info = nikkei_info[["Open", "Close"]]
    nikkei_info = nikkei_info.rename(columns={'Open': 'nikkei_open','Close': 'nikkei_close' })
    result.append(nikkei_info)

    # ドル円を取得する
    jpy = yf.Ticker("JPY=X")
    jpy_info = jpy.history(period="max",prepost=True,actions=False )
    jpy_info = jpy_info[["Open", "Close"]]
    jpy_info = jpy_info.rename(columns={'Open': 'jpy_open','Close': 'jpy_close' })
    result.append(jpy_info)


    # ニューヨークダウ平均株価を取得する
    dow = yf.Ticker("^DJI")
    dow_info = dow.history(period="max")
    dow_info = dow_info[["Open", "Close"]]
    dow_info = dow_info.rename(columns={'Open': 'dow_open','Close': 'dow_close' })
    result.append(dow_info)

    mini_dow = yf.Ticker("YM=F")
    mini_dow_info = mini_dow.history(period="max")
    mini_dow_info = mini_dow_info[["Open", "Close"]]
    mini_dow_info = mini_dow_info.rename(columns={'Open': 'mini_dow_open','Close': 'mini_dow_close' })
    result.append(mini_dow_info)
    

    # 配当金及び株式分割の確定日を取得
    result.append(company.actions)

    # # 財務諸表直近四年分
    # result.append(company.financials)

    # # 財務諸表直近四半期分取得
    # result.append(company.quarterly_financials)

    # # バランスシート直近4年分
    # result.append(company.balance_sheet)

    # # バランスシート直近四半期分
    # result.append(company.quarterly_balance_sheet)

    # 個別のデータフレームを1つのデータフレームにまとめデータを整形する
    return format.merge_all_company_info(result)


def get_prediction(code):
    """
    コードから明日の株価の予想をする

    Parameters:
    - code 株式コード
    Returns:
    - result 
    """
    try :
        # 企業情報を取得
        company = yf.Ticker(code+".T")
    except Exception as e :
        error_message = "企業データが存在しません、銘柄コードを確認してください"
        raise HTTPException(status_code=500, detail=error_message)

    try :

        # 分析に必要な株価財務データを取得
        datas = get_analysis_data(company)

        # 分析に必要な学習用、検証用データに分ける
        divided_datas = format.get_divided_data(datas)
        # 時系列交差分析を行う(検証時に利用)
        # scores = time_series_cross_analysis(divided_datas)
        price_prediction = price_predict(divided_datas)
    except Exception as e :
        error_message = "分析データの整形に失敗しました。再度試してください。"
        raise HTTPException(status_code=500, detail=error_message)

    # 予測を行う
    return {
        'prediction':price_prediction,
        'company':company.info.get('longName', 'No name found')
    }

def time_series_cross_analysis(divided_datas):
    """
    時系列交差分析を行いスコアを返す

    Parameters:
    - divided_datas 学習用、テスト用をそれぞれ目的変数、説明変数に分けたObject
    Returns:
    - valid_scores 予想結果制度スコア配列
    """

    valid_scores = []
    tscv = TimeSeriesSplit(n_splits=4)

    # 時系列分割交差検証
    for fold, (train_indices, valid_indices) in enumerate(tscv.split(divided_datas['X_train'])):
        X_train_cv, X_valid_cv = divided_datas['X_train'].iloc[train_indices], divided_datas['X_train'].iloc[valid_indices]
        Y_train_cv, Y_valid_cv = divided_datas['Y_train'].iloc[train_indices], divided_datas['Y_train'].iloc[valid_indices]

        # 線形回帰モデルのインスタンス化
        model = LinearRegression()

        # モデル学習
        model.fit(X_train_cv, Y_train_cv)

        # 予測
        Y_valid_pred = model.predict(X_valid_cv)

        # 予測精度(RMSE)の算出
        score = np.sqrt(mse(Y_valid_cv, Y_valid_pred))

        # 予測精度スコアをリストに格納
        valid_scores.append(score)

    return valid_scores


def price_predict(divided_datas):
    """
    重回帰分析により予測する

    Parameters:
    - divided_datas 学習用、テスト用をそれぞれ目的変数、説明変数に分けたObject
    Returns:
    - result 予想結果
    """

    # 説明変数の中で、X_train に存在するカラムのみを使用
    available_columns = [col for col in config.EXPLANATORY_VARIABLES_ANALYSIS if col in divided_datas['X_train'].columns]

    if not available_columns:
        raise ValueError("使用できる説明変数がありません。データの前処理を確認してください。")

    model = LinearRegression()

    # モデル学習
    model.fit(divided_datas['X_train'][available_columns], divided_datas['Y_train'])

    # テストデータにて予測する
    Y_pred = model.predict(divided_datas['X_test'][available_columns])

    # テストデータのスコアを取得（RMSE）
    score = np.sqrt(mse(divided_datas['Y_test'], Y_pred))

    # 過去の実際データを取得
    result = divided_datas['Y_test'].copy()
    result['Close_pred'] = Y_pred

    # 明日の株価を予測
    df = pd.DataFrame(divided_datas['X_test'])

    # 最終行の説明変数を取得
    last_data = divided_datas['last_data'].values.reshape(1,-1)
    tomorrow_prediction = model.predict(last_data)[0]

    # 翌営業日の日付を取得
    next_business_day = get_next_weekday(str(divided_datas['last_data'].name.strftime('%Y-%m-%d')))

    # 最新日のデータを追加
    last_row = pd.DataFrame({
        'Close_next': divided_datas['last_data']["Close"],
        'Close_pred': divided_datas['last_data']["Close"],
    }, index=[str(divided_datas['last_data'].name.strftime('%Y-%m-%d'))])

    # 予想の年月日のデータを追加
    new_row = pd.DataFrame({
        'Close_next': [0],
        'Close_pred': [tomorrow_prediction],
    }, index=[next_business_day])

    # インデックスを YYYY-MM-DD 形式に変更
    result.index = result.index.strftime('%Y-%m-%d')

    # 行を追加
    result = pd.concat([result, last_row])
    result = pd.concat([result, new_row])

    return {
        'close_next': result['Close_next'].to_dict(),
        'close_pred': result['Close_pred'].to_dict(),
        'score': float(score)
    }


def get_next_weekday(date_str):
    """
    翌営業日を取得する

    Parameters:
    - date_str: str (YYYY-MM-DD形式の日付)
    Returns:
    - result: str (YYYY-MM-DD形式の翌営業日)
    """

    # 文字列を datetime に変換
    date = datetime.strptime(date_str, '%Y-%m-%d')

    # 翌日を計算
    next_day = date + timedelta(days=1)

    # 土日なら次の平日まで進める
    while next_day.weekday() in [5, 6]:
        next_day += timedelta(days=1)

    return next_day.strftime('%Y-%m-%d')