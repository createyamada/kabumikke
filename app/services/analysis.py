from fastapi import HTTPException
# 線形回帰モデルのLinearRegressionをインポート
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
# 時系列分割のためTimeSeriesSplitのインポート
from sklearn.model_selection import TimeSeriesSplit
# 予測精度検証のためMSEをインポート
from sklearn.metrics import mean_squared_error as mse
# TDA is currently disabled in price prediction. Keep the dependency and
# implementation below so it can be restored after predictive validation.
# from ripser import ripser
import numpy as np
import pandas as pd
import re
import logging
import os
import pickle
import threading
from datetime import datetime
from datetime import timedelta
from pathlib import Path
from library import format
from library import config
from services import edinet
import yfinance as yf


logger = logging.getLogger(__name__)
HISTORY_PERIOD = "10y"

SECTOR_ETF_BY_KEYWORD = {
    'food': '1617.T', 'energy': '1618.T', 'oil': '1618.T',
    'construction': '1619.T', 'material': '1620.T', 'chemical': '1620.T',
    'pharma': '1621.T', 'healthcare': '1621.T', 'automotive': '1622.T',
    'transportation equipment': '1622.T', 'steel': '1623.T', 'metal': '1623.T',
    'machinery': '1624.T', 'electronic': '1625.T', 'semiconductor': '1625.T',
    'technology': '1626.T', 'communication': '1626.T', 'utilities': '1627.T',
    'transportation': '1628.T', 'logistics': '1628.T', 'trading': '1629.T',
    'wholesale': '1629.T', 'retail': '1630.T', 'bank': '1631.T',
    'financial': '1632.T', 'insurance': '1632.T', 'real estate': '1633.T',
}


def resolve_sector_benchmark(company):
    """Yahooの業種説明からTOPIX-17連動ETFを選び、業種ベンチマークにする。"""
    try:
        info = company.info or {}
    except Exception:
        return None, None
    description = ' '.join(str(info.get(key, '')).lower() for key in ('sector', 'industry', 'sectorKey', 'industryKey'))
    for keyword, symbol in SECTOR_ETF_BY_KEYWORD.items():
        if keyword in description:
            return symbol, info.get('industry') or info.get('sector')
    return None, info.get('industry') or info.get('sector')


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


def get_analysis_data(company, preloaded=None):
    """
    分析のためのデータを取得する

    Parameters:
    - company:str 特定の企業データ(yfinanceで取得)

    Returns:
    - result: 全分析データフレームの配列
    """
    result = []

    def supplied(key):
        frame = preloaded.get(key) if preloaded else None
        return frame.copy(deep=True) if isinstance(frame, pd.DataFrame) else None

    # 企業の株価時系列。空のまま市場指標だけを分析しないよう先に検証する。
    company_history = supplied('company')
    if company_history is None or company_history.empty:
        company_history = fetch_history(company, "company")
    result.append(company_history)


    # 日経平均株価を取得する
    nikkei_info = supplied('nikkei')
    if nikkei_info is None or nikkei_info.empty:
        nikkei = yf.Ticker("^N225")
        nikkei_info = fetch_history(nikkei, "nikkei", prepost=True, actions=False)
    nikkei_info = nikkei_info[["Open", "Close"]]
    nikkei_info = nikkei_info.rename(columns={'Open': 'nikkei_open','Close': 'nikkei_close' })
    result.append(nikkei_info)

    # 東証全体の地合いを表すTOPIX。取得失敗時は後段で日経平均を代理にする。
    topix_info = supplied('topix')
    if topix_info is None or topix_info.empty:
        topix = yf.Ticker("^TOPX")
        topix_info = fetch_history(topix, "topix", required=False, prepost=True, actions=False)
    if not topix_info.empty:
        topix_info = topix_info[["Open", "Close"]]
        topix_info = topix_info.rename(columns={"Open": "topix_open", "Close": "topix_close"})
    result.append(topix_info)

    sector_symbol = preloaded.get('sector_symbol') if preloaded else None
    sector_name = preloaded.get('sector_name') if preloaded else None
    sector_info = supplied('sector')
    if not preloaded:
        sector_symbol, sector_name = resolve_sector_benchmark(company)
    if sector_symbol:
        if sector_info is None or sector_info.empty:
            sector = yf.Ticker(sector_symbol)
            sector_info = fetch_history(sector, "sector_benchmark", required=False, actions=False)
        if not sector_info.empty:
            sector_info = sector_info[["Close"]].rename(columns={"Close": "sector_close"})
            result.append(sector_info)

    # ドル円を取得する
    jpy_info = supplied('jpy')
    if jpy_info is None or jpy_info.empty:
        jpy = yf.Ticker("JPY=X")
        jpy_info = fetch_history(jpy, "jpy", prepost=True, actions=False)
    jpy_info = jpy_info[["Open", "Close"]]
    jpy_info = jpy_info.rename(columns={'Open': 'jpy_open','Close': 'jpy_close' })
    result.append(jpy_info)


    # ニューヨークダウ平均株価を取得する
    dow_info = supplied('dow')
    if dow_info is None or dow_info.empty:
        dow = yf.Ticker("^DJI")
        dow_info = fetch_history(dow, "dow", actions=False)
    dow_info = dow_info[["Open", "Close"]]
    dow_info = dow_info.rename(columns={'Open': 'dow_open','Close': 'dow_close' })
    result.append(dow_info)

    mini_dow_info = supplied('mini_dow')
    if mini_dow_info is None or mini_dow_info.empty:
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
    merged = format.merge_all_company_info(result)
    merged.attrs['sector_benchmark_symbol'] = sector_symbol
    merged.attrs['sector_name'] = sector_name
    return merged


def _model_cache_path(code, market_date):
    root = Path(os.getenv('ANALYSIS_MODEL_CACHE_DIR', '.cache/models'))
    # v3: 米国市場・為替のpoint-in-time遅延、MASE、特徴量重要度を含む。
    return root / f'{code}_{market_date}_v3.joblib'


def _load_cached_prediction(code, market_date):
    if os.getenv('ANALYSIS_MODEL_CACHE_ENABLED', 'true').lower() not in {'1', 'true', 'yes'}:
        return None
    path = _model_cache_path(code, market_date)
    try:
        with path.open('rb') as cache_file:
            payload = pickle.load(cache_file)
        return payload.get('prediction') if payload.get('market_date') == market_date else None
    except (OSError, ValueError, EOFError, KeyError, pickle.UnpicklingError):
        return None


def _save_cached_prediction(code, market_date, prediction):
    if os.getenv('ANALYSIS_MODEL_CACHE_ENABLED', 'true').lower() not in {'1', 'true', 'yes'}:
        return
    path = _model_cache_path(code, market_date)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f'.{os.getpid()}.{threading.get_ident()}.tmp')
    with temporary.open('wb') as cache_file:
        pickle.dump({'market_date': market_date, 'prediction': prediction}, cache_file)
    os.replace(temporary, path)


def get_prediction(code, preloaded=None, company_name=None):
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
        datas = get_analysis_data(company, preloaded=preloaded)

        # 分析に必要な学習用、検証用データに分ける
        divided_datas = format.get_divided_data(datas)
        market_date = str(datas.index[-1].date())
        price_prediction = _load_cached_prediction(code, market_date)
        if price_prediction is None:
            price_prediction = price_predict(divided_datas)
            try:
                price_prediction['fundamental_analysis'] = edinet.get_fundamental_analysis(code)
            except Exception:
                logger.warning("EDINET analysis unavailable: code=%s", code, exc_info=True)
                price_prediction['fundamental_analysis'] = {
                    'available': False,
                    'source': 'EDINET_API_v2',
                    'reason': 'temporary_fetch_or_parse_error',
                }
            _save_cached_prediction(code, market_date, price_prediction)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e
    except Exception as e:
        logger.exception("stock analysis failed: code=%s", code)
        raise HTTPException(
            status_code=503,
            detail="株価データの取得または分析に失敗しました。時間をおいて再度試してください。",
        ) from e

    try:
        company_name = company_name or company.info.get('longName', code)
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


def compare_models_walk_forward(features, target, columns, horizon=1):
    """目的期間分をpurgeしたウォークフォワード検証でモデルを比較する。"""
    split_count = min(5, max(2, len(features) // 60))
    splitter = TimeSeriesSplit(n_splits=split_count, gap=max(1, int(horizon)))
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
            'purge_gap_business_days': int(horizon),
        }

    selected_name = min(comparison, key=lambda name: comparison[name]['walk_forward_rmse'])
    return selected_name, comparison


def calculate_permutation_importance(model, features, target, columns, top_n=15):
    """ホールドアウト誤差の増加量でモデル非依存の特徴量重要度を算出する。"""
    if features.empty:
        return []
    X = features[columns].copy()
    actual = np.asarray(target, dtype=float)
    baseline = float(mse(actual, model.predict(X)))
    random = np.random.default_rng(42)
    importance = []
    for column in columns:
        shuffled = X.copy()
        shuffled[column] = random.permutation(shuffled[column].to_numpy())
        shuffled_error = float(mse(actual, model.predict(shuffled)))
        importance.append({
            'feature': column,
            'mse_increase': float(shuffled_error - baseline),
        })
    importance.sort(key=lambda item: item['mse_increase'], reverse=True)
    return importance[:top_n]


def calculate_backtest(actual_returns, predicted_returns):
    """シグナルを1営業日遅延させ、コストとスリッページ込みで評価する。"""
    actual = np.asarray(actual_returns, dtype=float)
    predicted = np.asarray(predicted_returns, dtype=float)
    total_one_way_cost = config.TRANSACTION_COST_RATE + config.SLIPPAGE_RATE
    raw_signal = (predicted > total_one_way_cost).astype(float)
    # 終値確定後に生成したシグナルは、次の観測期間から有効にする。
    position = np.r_[0.0, raw_signal[:-1]]
    trades = np.abs(np.diff(np.r_[0.0, position]))
    gross_strategy_returns = position * actual
    strategy_returns = gross_strategy_returns - trades * total_one_way_cost
    strategy_curve = np.cumprod(1 + strategy_returns)
    gross_strategy_curve = np.cumprod(1 + gross_strategy_returns)
    buy_hold_curve = np.cumprod(1 + actual)
    running_peak = np.maximum.accumulate(strategy_curve)
    drawdown = strategy_curve / running_peak - 1
    volatility = np.std(strategy_returns, ddof=1)
    sharpe = np.sqrt(252) * np.mean(strategy_returns) / volatility if volatility else 0.0
    downside = strategy_returns[strategy_returns < 0]
    downside_volatility = np.std(downside, ddof=1) if len(downside) > 1 else 0.0
    sortino = (
        np.sqrt(252) * np.mean(strategy_returns) / downside_volatility
        if downside_volatility else 0.0
    )
    annualized_return = float(strategy_curve[-1] ** (252 / len(strategy_returns)) - 1)
    max_drawdown = float(np.min(drawdown))
    calmar = annualized_return / abs(max_drawdown) if max_drawdown else 0.0

    return {
        'strategy_return': float(strategy_curve[-1] - 1),
        'gross_strategy_return': float(gross_strategy_curve[-1] - 1),
        'buy_and_hold_return': float(buy_hold_curve[-1] - 1),
        'sharpe_ratio': float(sharpe),
        'sortino_ratio': float(sortino),
        'calmar_ratio': float(calmar),
        'annualized_return': annualized_return,
        'max_drawdown': max_drawdown,
        'trade_count': int(np.count_nonzero(trades)),
        'turnover': float(trades.sum()),
        'transaction_cost_rate': float(config.TRANSACTION_COST_RATE),
        'slippage_rate': float(config.SLIPPAGE_RATE),
        'total_estimated_cost': float(np.sum(trades) * total_one_way_cost),
        'signal_threshold': float(total_one_way_cost),
        'execution_lag_business_days': 1,
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
    # Lazy import prevents loading the TDA runtime during normal analysis.
    from ripser import ripser
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
        'predictive_validation': 'not_evaluated',
        'included_in_health_score': False,
    }


def analyze_topology_multi_window(divided_datas, windows=(60, 120, 252)):
    """複数期間のTDAと、直前窓からの複雑度変化を返す。"""
    results = {}
    for window in windows:
        try:
            current = analyze_topology(divided_datas, window=window)
            source = divided_datas['X_all'].iloc[:-max(10, window // 5)]
            previous_data = dict(divided_datas)
            previous_data['X_all'] = source
            previous_data['last_data'] = source.iloc[-1]
            previous = analyze_topology(previous_data, window=window)
            current['previous_loop_strength'] = previous['loop_strength']
            current['loop_strength_change'] = float(current['loop_strength'] - previous['loop_strength'])
            current['loop_strength_change_rate'] = float(
                current['loop_strength'] / max(previous['loop_strength'], 1e-12) - 1
            )
            results[str(window)] = current
        except (ValueError, IndexError):
            continue
    available = list(results.values())
    return {
        'windows': results,
        'trend': (
            'increasing_complexity' if available and np.mean([x['loop_strength_change'] for x in available]) > 0
            else 'decreasing_or_stable_complexity'
        ),
        'mean_loop_strength_change': float(np.mean([x['loop_strength_change'] for x in available])) if available else None,
    }


def predict_topix_excess_return(divided_datas, columns):
    """翌営業日の銘柄収益率－TOPIX収益率を独立した予測対象として学習する。"""
    source = divided_datas.get('source_data')
    features = divided_datas.get('feature_data')
    if source is None or features is None or 'topix_return' not in source:
        return {'available': False, 'reason': 'topix_data_unavailable'}
    stock_next = source['Close'].shift(-1) / source['Close'] - 1
    topix_next = source['topix_return'].shift(-1)
    labeled = pd.concat([features[columns], (stock_next - topix_next).rename('target')], axis=1).dropna()
    if len(labeled) < 80:
        return {'available': False, 'reason': 'insufficient_samples'}
    test_size = min(252, max(20, len(labeled) // 5))
    train = labeled.iloc[:-test_size]
    test = labeled.iloc[-test_size:]
    selected, comparison = compare_models_walk_forward(train, train['target'], columns, horizon=1)
    model = get_candidate_models()[selected]()
    model.fit(train[columns], train['target'])
    holdout = model.predict(test[columns])
    final_model = get_candidate_models()[selected]()
    final_model.fit(labeled[columns], labeled['target'])
    prediction = float(final_model.predict(features.iloc[-1][columns].to_frame().T)[0])
    actual = test['target'].to_numpy()
    return {
        'available': True,
        'target': 'next_day_stock_return_minus_topix_return',
        'predicted_excess_return': prediction,
        'selected_model': selected,
        'holdout_rmse': float(np.sqrt(mse(actual, holdout))),
        'directional_accuracy': float(np.mean(np.sign(actual) == np.sign(holdout))),
        'up_probability': estimate_up_probability(prediction, actual - holdout),
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


def evaluate_probability_calibration(actual_returns, predicted_returns, minimum_history=20):
    """過去の残差だけで逐次確率を作り、確率予測の校正状態を評価する。"""
    actual = np.asarray(actual_returns, dtype=float)
    predicted = np.asarray(predicted_returns, dtype=float)
    residuals = actual - predicted
    probabilities = []
    outcomes = []
    for index in range(minimum_history, len(actual)):
        probabilities.append(estimate_up_probability(predicted[index], residuals[:index]))
        outcomes.append(float(actual[index] > 0))

    if not probabilities:
        return {
            'method': 'expanding_residual_empirical_calibration',
            'sample_size': 0,
            'brier_score': None,
            'log_loss': None,
            'bins': [],
        }

    probabilities = np.asarray(probabilities)
    outcomes = np.asarray(outcomes)
    clipped = np.clip(probabilities, 1e-6, 1 - 1e-6)
    bins = []
    boundaries = np.linspace(0, 1, 6)
    for lower, upper in zip(boundaries[:-1], boundaries[1:]):
        mask = (probabilities >= lower) & (
            probabilities <= upper if upper == 1 else probabilities < upper
        )
        if np.any(mask):
            bins.append({
                'lower': float(lower),
                'upper': float(upper),
                'count': int(mask.sum()),
                'mean_predicted_probability': float(probabilities[mask].mean()),
                'observed_up_rate': float(outcomes[mask].mean()),
            })

    return {
        'method': 'expanding_residual_empirical_calibration',
        'sample_size': int(len(outcomes)),
        'brier_score': float(np.mean((probabilities - outcomes) ** 2)),
        'log_loss': float(-np.mean(outcomes * np.log(clipped) + (1 - outcomes) * np.log(1 - clipped))),
        'bins': bins,
    }


def train_calibrated_direction_classifier(X_train, y_train, X_test, y_test, last_data):
    """上昇専用ロジスティック分類器を、時系列順のIsotonic回帰で確率校正する。"""
    labels = (np.asarray(y_train) > 0).astype(int)
    if len(labels) < 60 or len(np.unique(labels)) < 2:
        return {'available': False, 'reason': 'insufficient_class_variation'}
    calibration_size = max(20, len(labels) // 5)
    fit_end = len(labels) - calibration_size
    if fit_end < 30 or len(np.unique(labels[:fit_end])) < 2:
        return {'available': False, 'reason': 'insufficient_training_samples'}

    classifier = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42),
    )
    classifier.fit(X_train.iloc[:fit_end], labels[:fit_end])
    raw_calibration = classifier.predict_proba(X_train.iloc[fit_end:])[:, 1]
    calibrator = IsotonicRegression(out_of_bounds='clip')
    calibrator.fit(raw_calibration, labels[fit_end:])
    raw_test = classifier.predict_proba(X_test)[:, 1]
    calibrated_test = np.asarray(calibrator.predict(raw_test), dtype=float)
    outcomes = (np.asarray(y_test) > 0).astype(float)
    last_raw = float(classifier.predict_proba(last_data)[0, 1])
    last_probability = float(calibrator.predict([last_raw])[0])
    clipped = np.clip(calibrated_test, 1e-6, 1 - 1e-6)
    return {
        'available': True,
        'model': 'logistic_regression',
        'calibration_method': 'isotonic_time_ordered_holdout',
        'up_probability': last_probability,
        'raw_up_probability': last_raw,
        'brier_score': float(np.mean((calibrated_test - outcomes) ** 2)),
        'log_loss': float(-np.mean(outcomes * np.log(clipped) + (1 - outcomes) * np.log(1 - clipped))),
        'directional_accuracy': float(np.mean((calibrated_test >= 0.5) == outcomes)),
        'test_samples': int(len(outcomes)),
    }


def calculate_return_risk(predicted_return, residuals, transaction_cost=None):
    """予測分布から期待値、損失確率、平均上下幅、リスクリワード比を算出する。"""
    cost = (config.TRANSACTION_COST_RATE + config.SLIPPAGE_RATE) if transaction_cost is None else transaction_cost
    errors = np.asarray(residuals, dtype=float)
    distribution = float(predicted_return) + errors[np.isfinite(errors)]
    if not len(distribution):
        return {'available': False}
    gains = distribution[distribution > 0]
    losses = distribution[distribution <= 0]
    mean_gain = float(gains.mean()) if len(gains) else 0.0
    mean_loss = float(abs(losses.mean())) if len(losses) else 0.0
    return {
        'available': True,
        'expected_return_after_cost': float(distribution.mean() - cost),
        'loss_probability': float(np.mean(distribution <= 0)),
        'gain_probability': float(np.mean(distribution > 0)),
        'average_gain': mean_gain,
        'average_loss': mean_loss,
        'reward_risk_ratio': float(mean_gain / mean_loss) if mean_loss else None,
        'expected_shortfall_10pct': float(np.mean(distribution[distribution <= np.quantile(distribution, 0.10)])),
        'distribution_quantiles': {
            key: float(np.quantile(distribution, quantile))
            for key, quantile in [('p10', 0.10), ('p25', 0.25), ('p50', 0.50), ('p75', 0.75), ('p90', 0.90)]
        },
    }


def build_adaptive_prediction_interval(
    actual_returns,
    predicted_returns,
    next_predicted_return,
    latest_close,
    target_coverage=config.PREDICTION_INTERVAL_COVERAGE,
    learning_rate=config.ADAPTIVE_CONFORMAL_LEARNING_RATE,
    minimum_history=20,
):
    """過去時点で利用可能な残差だけを使う適応的な非対称予測区間。"""
    actual = np.asarray(actual_returns, dtype=float)
    predicted = np.asarray(predicted_returns, dtype=float)
    residuals = actual - predicted
    alpha = 1 - target_coverage
    covered = []
    widths = []

    for index in range(minimum_history, len(actual)):
        history = residuals[:index]
        lower_error = float(np.quantile(history, alpha / 2))
        upper_error = float(np.quantile(history, 1 - alpha / 2))
        lower = predicted[index] + lower_error
        upper = predicted[index] + upper_error
        is_covered = lower <= actual[index] <= upper
        covered.append(is_covered)
        widths.append(upper - lower)
        error = 0.0 if is_covered else 1.0
        alpha = float(np.clip(alpha + learning_rate * ((1 - target_coverage) - error), 0.02, 0.50))

    lower_error = float(np.quantile(residuals, alpha / 2))
    upper_error = float(np.quantile(residuals, 1 - alpha / 2))
    lower_return = float(next_predicted_return + lower_error)
    upper_return = float(next_predicted_return + upper_error)
    interval = {
        'confidence': float(target_coverage),
        'method': 'adaptive_conformal_asymmetric_residual',
        'lower_return': lower_return,
        'upper_return': upper_return,
        'lower_price': float(latest_close * (1 + lower_return)),
        'upper_price': float(latest_close * (1 + upper_return)),
    }
    evaluation = {
        'target_coverage': float(target_coverage),
        'actual_coverage': float(np.mean(covered)) if covered else None,
        'evaluation_samples': int(len(covered)),
        'average_return_width': float(np.mean(widths)) if widths else None,
        'current_adaptive_alpha': float(alpha),
    }
    return interval, evaluation


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
        selected, comparison = compare_models_walk_forward(
            train,
            train['target'],
            columns,
            horizon=horizon,
        )
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


def build_confidence_assessment(
    metrics,
    comparison,
    selected_model,
    backtest,
    horizon_predictions,
    probability_evaluation,
    interval_evaluation,
):
    """検証済み指標を0～100点にまとめた分析健全性のヒューリスティック判定。"""
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
    brier_score = probability_evaluation.get('brier_score')
    if brier_score is not None and brier_score > 0.25:
        score -= 10
        reasons.append('上昇確率のBrier scoreが0.25を超過')

    actual_coverage = interval_evaluation.get('actual_coverage')
    target_coverage = interval_evaluation.get('target_coverage')
    if actual_coverage is not None and actual_coverage < target_coverage - 0.05:
        score -= 10
        reasons.append('予測区間の実被覆率が目標を5ポイント以上下回る')

    directions = [np.sign(item['predicted_return']) for item in horizon_predictions.values()]
    if directions and len(set(directions)) > 1:
        score -= 10
        reasons.append('予測期間によって上昇・下落方向が一致しない')

    score = int(max(0, min(100, score)))
    level = '高' if score >= 75 else '中' if score >= 50 else '低'
    signal = '候補' if score >= 75 else '監視' if score >= 50 else '見送り'
    return {
        'score_type': 'heuristic_analysis_health',
        'statistical_confidence': False,
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
            'probability_brier_score_max': 0.25,
            'prediction_interval_coverage_tolerance': 0.05,
        },
    }


def summarize_technical_analysis(divided_datas):
    """最新特徴量を、画面表示可能なシグナル・価格水準・0～100点へ整理する。"""
    source = divided_datas.get('source_data')
    if source is None or source.empty:
        return {'available': False}
    row = source.iloc[-1]

    def number(name, default=None):
        value = row.get(name, default)
        return float(value) if pd.notna(value) else default

    positive, negative, neutral = [], [], []
    score = 50
    if number('perfect_order_bull', 0) > 0:
        positive.append('移動平均線が上昇パーフェクトオーダー')
        score += 12
    elif number('perfect_order_bear', 0) > 0:
        negative.append('移動平均線が下降パーフェクトオーダー')
        score -= 12
    if number('golden_cross', 0) > 0:
        positive.append('当日にゴールデンクロスが発生')
        score += 8
    if number('dead_cross', 0) > 0:
        negative.append('当日にデッドクロスが発生')
        score -= 8
    if number('breakout_up20', 0) > 0:
        positive.append('20日高値を上抜け')
        score += 8
    if number('breakout_down20', 0) > 0:
        negative.append('20日安値を下抜け')
        score -= 8
    if number('higher_high', 0) > 0 and number('higher_low', 0) > 0:
        positive.append('高値・安値がともに切り上がり')
        score += 6
    adx = number('adx14')
    plus_di, minus_di = number('plus_di14'), number('minus_di14')
    if adx is not None and adx >= 0.25:
        if plus_di is not None and minus_di is not None and plus_di > minus_di:
            positive.append('ADXが示す上昇トレンドが強い')
            score += 6
        else:
            negative.append('ADXが示す下降トレンドが強い')
            score -= 6
    rsi = number('rsi14')
    if rsi is not None and rsi >= 0.70:
        negative.append('RSIが買われすぎ圏')
        score -= 3
    elif rsi is not None and rsi <= 0.30:
        neutral.append('RSIが売られすぎ圏で反発余地と下落継続の両面あり')
    cloud_position = number('ichimoku_cloud_position', 0)
    if cloud_position > 0:
        positive.append('株価が一目均衡表の雲より上')
        score += 5
    elif cloud_position < 0:
        negative.append('株価が一目均衡表の雲より下')
        score -= 5
    if number('bullish_engulfing', 0) > 0:
        positive.append('陽の包み足')
        score += 4
    if number('bearish_engulfing', 0) > 0:
        negative.append('陰の包み足')
        score -= 4
    if number('doji', 0) > 0:
        neutral.append('十字線に近く方向感が弱い')

    latest_close = number('Close', divided_datas.get('last_close'))
    resistance_gap = number('resistance_gap20')
    support_gap = number('support_gap20')
    resistance = latest_close / (1 + resistance_gap) if resistance_gap is not None and 1 + resistance_gap else None
    support = latest_close / (1 + support_gap) if support_gap is not None and 1 + support_gap else None
    return {
        'available': True,
        'technical_score': int(np.clip(score, 0, 100)),
        'signal': '強気' if score >= 65 else '弱気' if score <= 35 else '中立',
        'positive_factors': positive,
        'negative_factors': negative,
        'neutral_factors': neutral,
        'moving_average': {
            'sma5': number('SMA5'), 'sma25': number('SMA25'), 'sma70': number('SMA70'),
            'sma5_sma25_gap': number('sma5_sma25_gap'),
            'sma25_sma70_gap': number('sma25_sma70_gap'),
            'days_since_golden_cross': number('days_since_golden_cross'),
            'days_since_dead_cross': number('days_since_dead_cross'),
            'order_score': number('ma_order_score'),
        },
        'price_zone': {
            'resistance_20d': resistance, 'support_20d': support,
            'distance_from_high20': number('distance_from_high20'),
            'distance_from_high60': number('distance_from_high60'),
            'distance_from_high252': number('distance_from_high252'),
            'distance_from_low20': number('distance_from_low20'),
            'distance_from_low60': number('distance_from_low60'),
            'range_width20': number('range_width20'),
        },
        'candlestick': {
            'body_ratio': number('candle_body_ratio'),
            'upper_shadow_ratio': number('upper_shadow_ratio'),
            'lower_shadow_ratio': number('lower_shadow_ratio'),
            'gap_rate': number('gap_rate'),
            'consecutive_bullish': number('consecutive_bullish'),
            'consecutive_bearish': number('consecutive_bearish'),
            'bullish_engulfing': bool(number('bullish_engulfing', 0)),
            'bearish_engulfing': bool(number('bearish_engulfing', 0)),
            'doji': bool(number('doji', 0)),
        },
        'oscillators': {
            'rsi14': rsi, 'stochastic_k': number('stochastic_k'),
            'stochastic_d': number('stochastic_d'), 'adx14': adx,
            'plus_di14': plus_di, 'minus_di14': minus_di,
            'roc10': number('roc10'), 'cci20': number('cci20'), 'mfi14': number('mfi14'),
        },
        'volume': {
            'obv_slope20': number('obv_slope20'), 'vwap20_gap': number('vwap20_gap'),
            'volume_profile_poc_gap': number('volume_profile_poc_gap'),
            'volume_profile_value_area_width': number('volume_profile_value_area_width'),
        },
        'ichimoku': {
            'tenkan_gap': number('ichimoku_tenkan_gap'), 'kijun_gap': number('ichimoku_kijun_gap'),
            'tenkan_kijun_gap': number('ichimoku_tenkan_kijun_gap'),
            'cloud_position': cloud_position, 'cloud_width': number('ichimoku_cloud_width'),
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
        horizon=1,
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
    baseline_return_mae = np.mean(np.abs(actual.to_numpy() - baseline_pred))
    return_mase = return_mae / baseline_return_mae if baseline_return_mae else None
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

    feature_importance = calculate_permutation_importance(
        evaluation_model,
        divided_datas['X_test'],
        actual,
        available_columns,
    )

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

    # ホールドアウト残差を使い、時系列に追従する予測区間を作る。
    residuals = actual.to_numpy() - Y_pred
    prediction_interval, interval_evaluation = build_adaptive_prediction_interval(
        actual,
        Y_pred,
        tomorrow_return,
        latest_close,
    )

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
    # TDA calculation is intentionally disabled to reduce analysis time.
    # topology = analyze_topology(divided_datas)
    topology = None
    up_probability = estimate_up_probability(tomorrow_return, residuals)
    probability_evaluation = evaluate_probability_calibration(actual, Y_pred)
    classifier_probability = train_calibrated_direction_classifier(
        divided_datas['X_train'][available_columns],
        divided_datas['Y_train'],
        divided_datas['X_test'][available_columns],
        divided_datas['Y_test'],
        last_data,
    )
    if classifier_probability.get('available'):
        up_probability = classifier_probability['up_probability']
    return_risk = calculate_return_risk(tomorrow_return, residuals)
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
        'mase': float(return_mase) if return_mase is not None else None,
        'return_mase': float(return_mase) if return_mase is not None else None,
        'baseline_return_rmse': float(baseline_score),
        'baseline_return_mae': float(baseline_return_mae),
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
        horizon_predictions,
        probability_evaluation,
        interval_evaluation,
    )
    source_data = divided_datas.get('source_data')
    excess_return_prediction = predict_topix_excess_return(divided_datas, available_columns)
    # Multi-window TDA is retained above for future use but is not calculated.
    # topology_multi_window = analyze_topology_multi_window(divided_datas)
    topology_multi_window = None
    sector_source = (
        str(source_data['sector_benchmark_source'].iloc[-1])
        if source_data is not None and 'sector_benchmark_source' in source_data else None
    )
    industry_relative_strength = {
        'available': bool(source_data is not None and 'sector_relative_strength_20d' in source_data),
        'benchmark_symbol': source_data.attrs.get('sector_benchmark_symbol') if source_data is not None else None,
        'sector_name': source_data.attrs.get('sector_name') if source_data is not None else None,
        'benchmark_source': sector_source,
        'relative_strength_20d': (
            float(source_data['sector_relative_strength_20d'].iloc[-1])
            if source_data is not None and 'sector_relative_strength_20d' in source_data else None
        ),
    }
    data_quality = {
        'point_in_time_policy': 'japan_close_forecast_origin',
        'us_market_lag_business_days': 1,
        'daily_fx_lag_business_days': 1,
        'corporate_action_handling': 'yfinance_adjusted_history',
        'history_start': str(source_data.index.min().date()) if source_data is not None else None,
        'history_end': str(source_data.index.max().date()) if source_data is not None else None,
        'history_rows': int(len(source_data)) if source_data is not None else 0,
    }

    return {
        'close_next': result['Close_next'].to_dict(),
        'close_pred': result['Close_pred'].to_dict(),
        # 既存クライアント向けにscoreは従来どおり価格単位のRMSEとする。
        'score': float(price_score),
        'target': 'next_day_return',
        'selected_model': selected_model,
        'predicted_return': tomorrow_return,
        'up_probability': up_probability,
        'direction_classifier': classifier_probability,
        'return_risk': return_risk,
        'topix_excess_return_prediction': excess_return_prediction,
        'industry_relative_strength': industry_relative_strength,
        'technical_analysis': summarize_technical_analysis(divided_datas),
        'horizon_predictions': horizon_predictions,
        'confidence': confidence,
        'probability_evaluation': probability_evaluation,
        'prediction_interval': prediction_interval,
        'interval_evaluation': interval_evaluation,
        'model_comparison': model_comparison,
        'feature_importance': feature_importance,
        'backtest': backtest,
        'topological_analysis': topology,
        'topological_analysis_multi_window': topology_multi_window,
        'metrics': metrics,
        'data_quality': data_quality,
    }


def get_next_weekday(date_str):
    """
    翌営業日を取得する

    Parameters:
    - date_str: str (YYYY-MM-DD形式の日付)
    Returns:
    - result: str (YYYY-MM-DD形式の翌営業日)
    """

    date = pd.Timestamp(datetime.strptime(date_str, '%Y-%m-%d')).normalize()
    try:
        import exchange_calendars as xcals
        calendar = xcals.get_calendar('XTKS')
        sessions = calendar.sessions_in_range(
            date + pd.Timedelta(days=1),
            date + pd.Timedelta(days=14),
        )
        if len(sessions):
            return sessions[0].strftime('%Y-%m-%d')
    except (ImportError, KeyError, ValueError):
        logger.warning('JPX calendar unavailable; falling back to weekdays', exc_info=True)

    next_day = date + timedelta(days=1)
    while next_day.weekday() in [5, 6]:
        next_day += timedelta(days=1)
    return next_day.strftime('%Y-%m-%d')
