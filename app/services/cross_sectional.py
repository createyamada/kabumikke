"""Point-in-time cross-sectional backtest for a universe of TSE stocks."""
import numpy as np
import pandas as pd

from library import config


def run_cross_sectional_backtest(prices, top_n=10, rebalance_days=5, lookback_days=20):
    """過去モメンタム上位銘柄を翌日以降保有する、未来情報非混入の横断テスト。"""
    prices = pd.DataFrame(prices).sort_index().ffill()
    prices = prices.dropna(axis=1, thresh=max(30, len(prices) // 2))
    if len(prices) < lookback_days + rebalance_days + 20 or prices.shape[1] < 2:
        raise ValueError("cross-sectional backtest requires more history and at least two stocks")
    returns = prices.pct_change()
    momentum = prices.pct_change(lookback_days)
    weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    for position in range(lookback_days, len(prices), rebalance_days):
        signal = momentum.iloc[position].dropna().sort_values(ascending=False)
        selected = signal.head(min(top_n, len(signal))).index
        # シグナル算出翌日から保有し、同日終値の未来情報利用を避ける。
        start = position + 1
        end = min(position + 1 + rebalance_days, len(prices))
        if start < end and len(selected):
            weights.iloc[start:end, weights.columns.get_indexer(selected)] = 1.0 / len(selected)
    turnover = weights.diff().abs().sum(axis=1).fillna(weights.iloc[0].abs().sum())
    gross = (weights * returns).sum(axis=1)
    cost = turnover * (config.TRANSACTION_COST_RATE + config.SLIPPAGE_RATE)
    net = gross - cost
    equal_weight = returns.mean(axis=1).fillna(0.0)
    curve = (1 + net.fillna(0.0)).cumprod()
    benchmark_curve = (1 + equal_weight).cumprod()
    drawdown = curve / curve.cummax() - 1
    volatility = net.std(ddof=1)
    excess = net - equal_weight
    tracking_error = excess.std(ddof=1)
    return {
        "method": "lagged_cross_sectional_momentum",
        "universe_size": int(prices.shape[1]),
        "top_n": int(top_n),
        "lookback_business_days": int(lookback_days),
        "rebalance_business_days": int(rebalance_days),
        "strategy_return": float(curve.iloc[-1] - 1),
        "equal_weight_benchmark_return": float(benchmark_curve.iloc[-1] - 1),
        "excess_return": float(curve.iloc[-1] - benchmark_curve.iloc[-1]),
        "sharpe_ratio": float(np.sqrt(252) * net.mean() / volatility) if volatility else 0.0,
        "information_ratio": float(np.sqrt(252) * excess.mean() / tracking_error) if tracking_error else 0.0,
        "max_drawdown": float(drawdown.min()),
        "turnover": float(turnover.sum()),
        "estimated_cost": float(cost.sum()),
        "point_in_time_policy": "signal_at_close_execute_next_business_day",
    }


def fetch_and_run_cross_sectional_backtest(codes, period="5y", top_n=10, rebalance_days=5):
    import yfinance as yf

    symbols = [f"{str(code).strip()[:4]}.T" for code in codes]
    downloaded = yf.download(symbols, period=period, auto_adjust=True, progress=False)
    close = downloaded["Close"] if isinstance(downloaded.columns, pd.MultiIndex) else downloaded
    if isinstance(close, pd.Series):
        close = close.to_frame()
    result = run_cross_sectional_backtest(close, top_n=top_n, rebalance_days=rebalance_days)
    result["codes"] = [symbol.removesuffix(".T") for symbol in close.columns]
    return result
