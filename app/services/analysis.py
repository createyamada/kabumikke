from fastapi import HTTPException
# 線形回帰モデルのLinearRegressionをインポート
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
# 時系列分割のためTimeSeriesSplitのインポート
from sklearn.model_selection import TimeSeriesSplit
# 予測精度検証のためMSEをインポート
from sklearn.metrics import mean_squared_error as mse
import numpy as np
import pandas as pd
import re
import logging
from datetime import datetime
from datetime import timedelta
from library import format
from library import config
import yfinance as yf


logger = logging.getLogger(__name__)
HISTORY_PERIOD = "10y"


def fetch_history(ticker, label, required=True, **kwargs):
    """yfinanceの一時的な取得失敗を1回だけ再試行する。"""
    last_error = None
    for attempt in range(2):
        try:
            history = ticker.history(period=HISTORY_PERIOD, **kwargs)
            if history is not None and not history.empty:
                return history
            last_error = RuntimeError(f"{label}の履歴が空です。")
        except Exception as error:
            last_error = error
            logger.warning(
                "market data fetch failed: label=%s attempt=%s",
                label,
                attempt + 1,
                exc_info=True,
            )

    if required:
        raise RuntimeError(f"{label}の市場データを取得できませんでした。") from last_error
    logger.warning("optional market data unavailable: label=%s", label)
    return pd.DataFrame()


def get_analysis_data(company):
    """
    分析のためのデータを取得する

    Parameters:
    - company:str 特定の企業データ(yfinanceで取得)

    Returns:
    - result: 全分析データフレームの配列
    """
    result = []

    # 企業の株価時系列。空のまま市場指標だけを分析しないよう先に検証する。
    company_history = fetch_history(company, "company")
    result.append(company_history)


    # 日経平均株価を取得する
    nikkei = yf.Ticker("^N225")
    nikkei_info = fetch_history(nikkei, "nikkei", prepost=True, actions=False)
    nikkei_info = nikkei_info[["Open", "Close"]]
    nikkei_info = nikkei_info.rename(columns={'Open': 'nikkei_open','Close': 'nikkei_close' })
    result.append(nikkei_info)

    # ドル円を取得する
    jpy = yf.Ticker("JPY=X")
    jpy_info = fetch_history(jpy, "jpy", prepost=True, actions=False)
    jpy_info = jpy_info[["Open", "Close"]]
    jpy_info = jpy_info.rename(columns={'Open': 'jpy_open','Close': 'jpy_close' })
    result.append(jpy_info)


    # ニューヨークダウ平均株価を取得する
    dow = yf.Ticker("^DJI")
    dow_info = fetch_history(dow, "dow", actions=False)
    dow_info = dow_info[["Open", "Close"]]
    dow_info = dow_info.rename(columns={'Open': 'dow_open','Close': 'dow_close' })
    result.append(dow_info)

    mini_dow = yf.Ticker("YM=F")
    mini_dow_info = fetch_history(mini_dow, "mini_dow", required=False, actions=False)
    if not mini_dow_info.empty:
        mini_dow_info = mini_dow_info[["Open", "Close"]]
        mini_dow_info = mini_dow_info.rename(columns={'Open': 'mini_dow_open','Close': 'mini_dow_close' })
        result.append(mini_dow_info)
    

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
    code = code.strip()
    if not re.fullmatch(r"\d{4}", code):
        raise HTTPException(status_code=422, detail="銘柄コードは4桁の数字で指定してください。")

    company = yf.Ticker(code + ".T")

    try :

        # 分析に必要な株価財務データを取得
        datas = get_analysis_data(company)

        # 分析に必要な学習用、検証用データに分ける
        divided_datas = format.get_divided_data(datas)
        price_prediction = price_predict(divided_datas)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e
    except Exception as e:
        logger.exception("stock analysis failed: code=%s", code)
        raise HTTPException(
            status_code=503,
            detail="株価データの取得または分析に失敗しました。時間をおいて再度試してください。",
        ) from e

    try:
        company_name = company.info.get('longName', code)
    except Exception:
        # 会社名は表示用なので、取得失敗で分析結果を失わないようにする。
        company_name = code

    # 予測を行う
    return {
        'prediction':price_prediction,
        'company': company_name,
    }

def get_candidate_models():
    """比較対象モデルを毎回新しいインスタンスで返す。"""
    return {
        'linear_regression': lambda: make_pipeline(StandardScaler(), LinearRegression()),
        'ridge': lambda: make_pipeline(StandardScaler(), Ridge(alpha=10.0)),
        'gradient_boosting': lambda: HistGradientBoostingRegressor(
            learning_rate=0.05,
            max_iter=200,
            max_leaf_nodes=15,
            l2_regularization=1.0,
            random_state=42,
        ),
    }


def compare_models_walk_forward(features, target, columns):
    """過去から未来へ進む交差検証でモデルを比較する。"""
    split_count = min(5, max(2, len(features) // 60))
    splitter = TimeSeriesSplit(n_splits=split_count)
    comparison = {}

    for name, factory in get_candidate_models().items():
        fold_scores = []
        for train_indices, valid_indices in splitter.split(features):
            model = factory()
            model.fit(features.iloc[train_indices][columns], target.iloc[train_indices])
            prediction = model.predict(features.iloc[valid_indices][columns])
            fold_scores.append(float(np.sqrt(mse(target.iloc[valid_indices], prediction))))
        comparison[name] = {
            'walk_forward_rmse': float(np.mean(fold_scores)),
            'fold_scores': fold_scores,
        }

    selected_name = min(comparison, key=lambda name: comparison[name]['walk_forward_rmse'])
    return selected_name, comparison


def calculate_backtest(actual_returns, predicted_returns):
    """ロング・現金戦略を、売買コスト込みで評価する。"""
    actual = np.asarray(actual_returns, dtype=float)
    predicted = np.asarray(predicted_returns, dtype=float)
    # 期待収益が片道コストを上回る場合だけ保有する。
    position = (predicted > config.TRANSACTION_COST_RATE).astype(float)
    trades = np.abs(np.diff(np.r_[0.0, position]))
    strategy_returns = position * actual - trades * config.TRANSACTION_COST_RATE
    strategy_curve = np.cumprod(1 + strategy_returns)
    buy_hold_curve = np.cumprod(1 + actual)
    running_peak = np.maximum.accumulate(strategy_curve)
    drawdown = strategy_curve / running_peak - 1
    volatility = np.std(strategy_returns, ddof=1)
    sharpe = np.sqrt(252) * np.mean(strategy_returns) / volatility if volatility else 0.0

    return {
        'strategy_return': float(strategy_curve[-1] - 1),
        'buy_and_hold_return': float(buy_hold_curve[-1] - 1),
        'sharpe_ratio': float(sharpe),
        'max_drawdown': float(np.min(drawdown)),
        'trade_count': int(np.count_nonzero(trades)),
        'transaction_cost_rate': float(config.TRANSACTION_COST_RATE),
        'signal_threshold': float(config.TRANSACTION_COST_RATE),
    }


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

    selected_model, model_comparison = compare_models_walk_forward(
        divided_datas['X_train'],
        divided_datas['Y_train'],
        available_columns,
    )
    evaluation_model = get_candidate_models()[selected_model]()
    evaluation_model.fit(
        divided_datas['X_train'][available_columns],
        divided_datas['Y_train'],
    )

    # テストデータにて予測する
    Y_pred = evaluation_model.predict(divided_datas['X_test'][available_columns])

    # 収益率として評価する。ゼロは「価格変化なし」の単純予測。
    actual = divided_datas['Y_test']
    return_score = np.sqrt(mse(actual, Y_pred))
    return_mae = np.mean(np.abs(actual.to_numpy() - Y_pred))
    baseline_pred = np.zeros(len(actual))
    baseline_score = np.sqrt(mse(actual, baseline_pred))
    actual_direction = np.sign(actual.to_numpy())
    predicted_direction = np.sign(Y_pred)
    directional_accuracy = np.mean(actual_direction == predicted_direction)

    for name, factory in get_candidate_models().items():
        holdout_model = factory()
        holdout_model.fit(
            divided_datas['X_train'][available_columns],
            divided_datas['Y_train'],
        )
        holdout_prediction = holdout_model.predict(divided_datas['X_test'][available_columns])
        model_comparison[name]['holdout_rmse'] = float(np.sqrt(mse(actual, holdout_prediction)))

    current_close = divided_datas['current_close_test'].to_numpy()
    predicted_close = current_close * (1 + Y_pred)
    result = divided_datas['actual_close_test'].to_frame(name='Close_next')
    result['Close_pred'] = predicted_close
    price_score = np.sqrt(mse(result['Close_next'], predicted_close))
    price_mae = np.mean(np.abs(result['Close_next'].to_numpy() - predicted_close))
    baseline_price_score = np.sqrt(mse(result['Close_next'], current_close))

    # 最終行の説明変数を取得
    # 評価用データも含む全履歴で学習し直し、最新情報を翌営業日予測に反映する。
    final_model = get_candidate_models()[selected_model]()
    final_model.fit(
        divided_datas['X_all'][available_columns],
        divided_datas['Y_all'],
    )
    last_data = divided_datas['last_data'][available_columns].to_frame().T
    tomorrow_return = float(final_model.predict(last_data)[0])
    latest_close = divided_datas['last_close']
    tomorrow_prediction = latest_close * (1 + tomorrow_return)

    # ホールドアウト残差から外れ値に頑健な80%予測区間を作る。
    residuals = actual.to_numpy() - Y_pred
    lower_return = tomorrow_return + float(np.quantile(residuals, 0.10))
    upper_return = tomorrow_return + float(np.quantile(residuals, 0.90))
    prediction_interval = {
        'confidence': 0.80,
        'lower_return': lower_return,
        'upper_return': upper_return,
        'lower_price': latest_close * (1 + lower_return),
        'upper_price': latest_close * (1 + upper_return),
    }

    # 翌営業日の日付を取得
    next_business_day = get_next_weekday(str(divided_datas['last_data'].name.strftime('%Y-%m-%d')))

    # 最新日のデータを追加
    last_row = pd.DataFrame({
        'Close_next': latest_close,
        'Close_pred': latest_close,
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
        # 既存クライアント向けにscoreは従来どおり価格単位のRMSEとする。
        'score': float(price_score),
        'target': 'next_day_return',
        'selected_model': selected_model,
        'predicted_return': tomorrow_return,
        'prediction_interval': prediction_interval,
        'model_comparison': model_comparison,
        'backtest': calculate_backtest(actual, Y_pred),
        'metrics': {
            'rmse': float(price_score),
            'mae': float(price_mae),
            'baseline_rmse': float(baseline_price_score),
            'return_rmse': float(return_score),
            'return_mae': float(return_mae),
            'baseline_return_rmse': float(baseline_score),
            'rmse_improvement_rate': (
                float(1 - return_score / baseline_score) if baseline_score else 0.0
            ),
            'directional_accuracy': float(directional_accuracy),
            'test_samples': int(len(actual)),
            'training_samples': int(len(divided_datas['Y_all'])),
        },
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
