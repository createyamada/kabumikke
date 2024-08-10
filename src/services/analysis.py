
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



def get_company_info(code_ :str):
    """
    特定コードの企業情報を取得する

    Parameters:
    - code_:str 株価コード
    Returns:
    - result: object 企業情報
    """

    return yf.Ticker(code_+".T")


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

    # 配当金及び株式分割の確定日を取得
    result.append(company.actions)

    # # 財務諸表直近四年分
    result.append(company.financials)

    # # 財務諸表直近四半期分取得
    result.append(company.quarterly_financials)

    # # バランスシート直近4年分
    result.append(company.balance_sheet)

    # # バランスシート直近四半期分
    result.append(company.quarterly_balance_sheet)

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


    # 企業情報を取得
    company = get_company_info(code)
    # 分析に必要な株価財務データを取得
    datas = get_analysis_data(company)
    # 分析に必要な学習用、検証用データに分ける
    divided_datas = format.get_divided_data(datas)
    # 時系列交差分析を行う(検証時に利用)
    # scores = time_series_cross_analysis(divided_datas)

    # 予測を行う
    return {
        'prediction':price_predict(divided_datas),
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



    model = LinearRegression()
    # 説明変数、目的変数を指定
    model.fit(divided_datas['X_train'], divided_datas['Y_train'])

    # テストデータにて予測する
    Y_pred = model.predict(divided_datas['X_test'])

    # 予測モデルの係数を確認
    # coef = pd.DataFrame(model.coef_)
    # coef.index = divided_datas['X_train'].columns

    # 予測モデルの切片を取得
    # model.intercept_

    # X_train基本統計量確認
    # divided_datas['X_train'].discribe()

    # テストデータのスコアを取得
    score = np.sqrt(mse(divided_datas['Y_test'], Y_pred))

    # 過去の実際データを取得
    result = divided_datas['Y_test']
    result['Close_pred'] = Y_pred

    # 明日の株価を予測
    df = pd.DataFrame(divided_datas['X_test'])
    last_data = df[config.EXPLANATORY_VARIABLES_ANALYSIS].iloc[-1].values.reshape(1, -1)
    tomorrow_prediction = model.predict(last_data)
    
    # データフレームに追加する新しい行を作成
    new_row = pd.DataFrame({
        'Close_next': 0,
        'Close_pred': tomorrow_prediction[0],
    },index=[get_next_weekday(str(result.index[-1].strftime('%Y-%m-%d')))])

    # インデックスをYYYY-MM-DD形式の文字列に変換
    result.index = result.index.strftime('%Y-%m-%d')
    # 行を追加
    result = pd.concat([result, new_row])

    return {
        'close_next':result['Close_next'].to_dict(),
        'close_pred':result['Close_pred'].to_dict(),
        'score':np.mean(score)
    }



def get_next_weekday(date_str):
    """
    翌営業日を取得する

    Parameters:
    - date 日付
    Returns:
    - result 翌営業日
    """

    # 文字列を datetime オブジェクトに変換
    date = datetime.strptime(date_str, '%Y-%m-%d')
    
    # 次の日を計算
    next_day = date + timedelta(days=1)
    
    # 次の日が土日であるか確認
    while next_day.weekday() in [5, 6]:  # 5 = 土曜日, 6 = 日曜日
        next_day += timedelta(days=1)
    
    # YYYY-MM-DD 形式の文字列に変換
    return next_day.strftime('%Y-%m-%d')


