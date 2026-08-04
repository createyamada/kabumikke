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
from ripser import ripser
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

    # 東証全体の地合いを表すTOPIX。取得失敗時は後段で日経平均を代理にする。
    topix = yf.Ticker("^TOPX")
    topix_info = fetch_history(topix, "topix", required=False, prepost=True, actions=False)
    if not topix_info.empty:
        topix_info = topix_info[["Open", "Close"]]
        topix_info = topix_info.rename(columns={"Open": "topix_open", "Close": "topix_close"})
        result.append(topix_info)

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


def create_delay_embedding(values, dimension=3, delay=1):
    """1次元の時系列をTakens型の遅延座標へ埋め込む。"""
    series = np.asarray(values, dtype=float)
    series = series[np.isfinite(series)]
    required = (dimension - 1) * delay + 1
    if len(series) < max(40, required):
        raise ValueError("トポロジカル分析に必要な履歴が不足しています。")

    point_count = len(series) - (dimension - 1) * delay
    point_cloud = np.column_stack([
        series[offset * delay:offset * delay + point_count]
        for offset in range(dimension)
    ])
    mean = point_cloud.mean(axis=0)
    standard_deviation = point_cloud.std(axis=0)
    standard_deviation[standard_deviation == 0] = 1.0
    return (point_cloud - mean) / standard_deviation


def summarize_persistence_diagram(diagram):
    """永続図をJSON化しやすい集約指標へ変換する。"""
    intervals = np.asarray(diagram, dtype=float)
    if intervals.size == 0:
        lifetimes = np.array([], dtype=float)
    else:
        finite = intervals[np.isfinite(intervals[:, 1])]
        lifetimes = finite[:, 1] - finite[:, 0]
        lifetimes = lifetimes[lifetimes > 1e-9]

    total = float(lifetimes.sum())
    if total:
        probabilities = lifetimes / total
        entropy = float(-np.sum(probabilities * np.log(probabilities)))
    else:
        entropy = 0.0

    return {
        'feature_count': int(len(lifetimes)),
        'total_persistence': total,
        'max_persistence': float(lifetimes.max()) if len(lifetimes) else 0.0,
        'mean_persistence': float(lifetimes.mean()) if len(lifetimes) else 0.0,
        'persistence_entropy': entropy,
    }


def analyze_topology(divided_datas, window=252, dimension=3, delay=1):
    """株価リターン点群のH0/H1パーシステントホモロジーを集約する。"""
    returns = divided_datas['X_all']['return_1d'].tail(window).to_numpy()
    latest_return = divided_datas['last_data'].get('return_1d')
    if pd.notna(latest_return):
        returns = np.r_[returns, float(latest_return)][-window:]

    point_cloud = create_delay_embedding(returns, dimension=dimension, delay=delay)
    diagrams = ripser(point_cloud, maxdim=1)['dgms']
    h0 = summarize_persistence_diagram(diagrams[0])
    h1 = summarize_persistence_diagram(diagrams[1])

    # H0に対するH1の永続量比を、市場構造の複雑さを示す補助指標として扱う。
    loop_strength = h1['total_persistence'] / max(h0['total_persistence'], 1e-12)
    if loop_strength < 0.05:
        regime = 'low_topological_complexity'
    elif loop_strength < 0.15:
        regime = 'moderate_topological_complexity'
    else:
        regime = 'high_topological_complexity'

    return {
        'method': 'vietoris_rips_persistent_homology',
        'role': 'market_structure_indicator_not_price_forecast',
        'source': 'daily_return_delay_embedding',
        'sample_size': int(len(returns)),
        'point_count': int(len(point_cloud)),
        'embedding_dimension': int(dimension),
        'delay': int(delay),
        'h0_connected_components': h0,
        'h1_loops': h1,
        'loop_strength': float(loop_strength),
        'regime': regime,
        'interpretation': 'heuristic',
    }


def estimate_up_probability(predicted_return, residuals):
    """検証期間の誤差分布から、実現収益率が0を超える経験確率を求める。"""
    errors = np.asarray(residuals, dtype=float)
    errors = errors[np.isfinite(errors)]
    if not len(errors):
        return 0.5
    # actual = prediction + residual とみなし、ラプラス補正で0/1への張り付きを防ぐ。
    favorable = np.count_nonzero(errors > -float(predicted_return))
    return float((favorable + 1) / (len(errors) + 2))


def predict_multiple_horizons(divided_datas, columns, horizons=(5, 20)):
    """同じ特徴量から5・20営業日先の収益率と上昇確率を時系列検証付きで返す。"""
    source = divided_datas.get('source_data')
    features = divided_datas.get('feature_data')
    if source is None or features is None:
        return {}

    forecasts = {}
    latest_features = features.iloc[-1][columns].to_frame().T
    for horizon in horizons:
        target = source['Close'].shift(-horizon) / source['Close'] - 1
        labeled = pd.concat([features[columns], target.rename('target')], axis=1).dropna()
        if len(labeled) < 80:
            continue
        test_size = min(252, max(20, len(labeled) // 5))
        # 予測期間分を境界から除外し、学習ラベルが評価期間へはみ出す未来情報混入を防ぐ。
        train_end = len(labeled) - test_size - horizon
        if train_end < 40:
            continue
        train = labeled.iloc[:train_end]
        test = labeled.iloc[-test_size:]
        selected, comparison = compare_models_walk_forward(train, train['target'], columns)
        model = get_candidate_models()[selected]()
        model.fit(train[columns], train['target'])
        holdout_prediction = model.predict(test[columns])
        residuals = test['target'].to_numpy() - holdout_prediction
        final_model = get_candidate_models()[selected]()
        final_model.fit(labeled[columns], labeled['target'])
        predicted_return = float(final_model.predict(latest_features)[0])
        forecasts[str(horizon)] = {
            'horizon_business_days': int(horizon),
            'predicted_return': predicted_return,
            'predicted_price': float(divided_datas['last_close'] * (1 + predicted_return)),
            'up_probability': estimate_up_probability(predicted_return, residuals),
            'selected_model': selected,
            'holdout_return_rmse': float(np.sqrt(mse(test['target'], holdout_prediction))),
            'walk_forward_return_rmse': float(comparison[selected]['walk_forward_rmse']),
        }
    return forecasts


def build_confidence_assessment(metrics, comparison, selected_model, backtest, topology, horizon_predictions):
    """予測精度・運用成績・相場構造を0～100点に統合する説明可能な判定。"""
    reasons = []
    score = 100
    improvement = metrics['rmse_improvement_rate']
    direction = metrics['directional_accuracy']
    selected = comparison[selected_model]
    walk_forward = max(selected['walk_forward_rmse'], 1e-12)
    stability_ratio = selected['holdout_rmse'] / walk_forward

    if improvement < 0.05:
        score -= 20
        reasons.append('ベースライン比較改善率が5%未満')
    if direction < 0.55:
        score -= 20
        reasons.append('方向一致率が55%未満')
    if stability_ratio > 1.25:
        score -= 20
        reasons.append('直近データのRMSEがウォークフォワードRMSEから乖離')
    if backtest['sharpe_ratio'] < 1.0:
        score -= 15
        reasons.append('シャープレシオが1未満')
    if backtest['strategy_return'] <= backtest['buy_and_hold_return']:
        score -= 15
        reasons.append('戦略リターンが買い持ちリターン以下')
    if topology['regime'] == 'high_topological_complexity':
        score -= 20
        reasons.append('TDAが高トポロジー複雑度を検出')
    elif topology['regime'] == 'moderate_topological_complexity':
        score -= 5
        reasons.append('TDAが中程度の市場複雑度を検出')

    directions = [np.sign(item['predicted_return']) for item in horizon_predictions.values()]
    if directions and len(set(directions)) > 1:
        score -= 10
        reasons.append('予測期間によって上昇・下落方向が一致しない')

    score = int(max(0, min(100, score)))
    level = '高' if score >= 75 else '中' if score >= 50 else '低'
    signal = '候補' if score >= 75 else '監視' if score >= 50 else '見送り'
    return {
        'confidence_score': score,
        'confidence_level': level,
        'trade_signal': signal,
        'risk_reasons': reasons,
        'holdout_to_walk_forward_rmse_ratio': float(stability_ratio),
        'criteria': {
            'rmse_improvement_rate_min': 0.05,
            'directional_accuracy_min': 0.55,
            'rmse_stability_ratio_max': 1.25,
            'sharpe_ratio_min': 1.0,
            'strategy_must_beat_buy_and_hold': True,
            'high_topological_complexity_allowed': False,
        },
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

    backtest = calculate_backtest(actual, Y_pred)
    topology = analyze_topology(divided_datas)
    up_probability = estimate_up_probability(tomorrow_return, residuals)
    horizon_predictions = {
        '1': {
            'horizon_business_days': 1,
            'predicted_return': tomorrow_return,
            'predicted_price': float(tomorrow_prediction),
            'up_probability': up_probability,
            'selected_model': selected_model,
            'holdout_return_rmse': float(return_score),
            'walk_forward_return_rmse': float(model_comparison[selected_model]['walk_forward_rmse']),
        }
    }
    horizon_predictions.update(
        predict_multiple_horizons(divided_datas, available_columns, horizons=(5, 20))
    )
    metrics = {
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
    }
    confidence = build_confidence_assessment(
        metrics,
        model_comparison,
        selected_model,
        backtest,
        topology,
        horizon_predictions,
    )

    return {
        'close_next': result['Close_next'].to_dict(),
        'close_pred': result['Close_pred'].to_dict(),
        # 既存クライアント向けにscoreは従来どおり価格単位のRMSEとする。
        'score': float(price_score),
        'target': 'next_day_return',
        'selected_model': selected_model,
        'predicted_return': tomorrow_return,
        'up_probability': up_probability,
        'horizon_predictions': horizon_predictions,
        'confidence': confidence,
        'prediction_interval': prediction_interval,
        'model_comparison': model_comparison,
        'backtest': backtest,
        'topological_analysis': topology,
        'metrics': metrics,
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
