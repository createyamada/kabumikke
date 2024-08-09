
# 線形回帰モデルのLinearRegressionをインポート
from sklearn.linear_model import LinearRegression
# 時系列分割のためTimeSeriesSplitのインポート
from sklearn.model_selection import TimeSeriesSplit
# 予測精度検証のためMSEをインポート
from sklearn.metrics import mean_squared_error as mse
import numpy as np
import pandas as pd
from library import finance
from library import format


def get_prediction(code):
    """
    コードから明日の株価の予想をする

    Parameters:
    - code 株式コード
    Returns:
    - result 
    """


    # 企業情報を取得
    company = finance.get_company_info(code)
    # 分析に必要な株価財務データを取得
    datas = finance.get_analysis_data(company)
    # 分析に必要な学習用、検証用データに分ける
    divided_datas = format.get_divided_data(datas)
    # 時系列交差分析を行う
    # scores = time_series_cross_analysis(divided_datas)
    # 予測を行う
    # print(divided_datas)
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
    価格を重回帰分析により予測する

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
    coef = pd.DataFrame(model.coef_)
    coef.index = divided_datas['X_train'].columns

    # 予測モデルの切片を取得
    # model.intercept_

    # X_train基本統計量確認
    # divided_datas['X_train'].discribe()

    # テストデータのスコアを取得
    score = np.sqrt(mse(divided_datas['Y_test'], Y_pred))

    # 過去の実際データを取得
    result = divided_datas['Y_test']
    df = pd.DataFrame(result)
    df['Close_pred'] = Y_pred

    # データの形式を確認
    print("Data types:")
    print(df.dtypes)
    print(df)


    return {
        'close_next':df['Close_next'].to_dict(),
        'close_pred':df['Close_pred'].to_dict(),
        'score':np.mean(score)
    }


