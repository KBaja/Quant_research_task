import numpy as np
import pandas as pd



def backtest_strategy(
    df,
    sma_col="sma",
    price_col="mid_price",
    signal_col="signal",
    initial_capital=100_000.0,
    allow_short_enter=False,
    transaction_cost_pct=0.0002,  # 0.02% per side
    verbose=True
):
    """
    Simulate the SMA strategy with optional control of the first entry side and
    realistic transaction costs applied per side.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain price, SMA derived signal, and an index in time order.
    sma_col : str
        Column name for SMA. Kept for context.
    price_col : str
        Column name for price.
    signal_col : str
        Column name for precomputed signal where +1 is long and -1 is short.
    initial_capital : float
        Starting capital in USD.
    allow_short_enter : bool
        If False, remain flat until the first long appears. If True, may start short.
    transaction_cost_pct : float
        Proportional cost per executed side. For example 0.0002 equals 0.02 percent.
    """

    df = df.copy()

    # Use provided signals and create a filled version that persists until a flip
    raw_signal = df[signal_col].astype(float)
    signal_filled = raw_signal.replace(0, np.nan).ffill()

    # Build position with an in position flag and the allow_short_enter rule
    position = []
    in_position = False
    current_pos = 0.0

    for s in signal_filled:
        if pd.isna(s) or s == 0:
            pos = current_pos if in_position else 0.0
        else:
            if not in_position:
                if s == -1.0 and not allow_short_enter:
                    pos = 0.0
                else:
                    current_pos = float(s)
                    in_position = True
                    pos = current_pos
            else:
                if s != current_pos:
                    current_pos = float(s)
                pos = current_pos
        position.append(pos)

    df["position"] = position

    # Compute asset returns
    df["asset_return"] = df[price_col].pct_change().fillna(0.0)

    # Strategy returns from the position held over the interval
    df["strategy_return_gross"] = df["position"].shift(1).fillna(0.0) * df["asset_return"]

    # Transaction costs per bar
    # Count executed sides from position changes
    # Entry from 0 to +1 or -1 counts as 1 side
    # Flip from +1 to -1 or reverse counts as 2 sides
    pos_diff = df["position"].diff().fillna(0.0).abs()
    executed_sides = pos_diff  # values are 0, 1, or 2 with positions in {-1, 0, +1}
    df["tx_cost"] = executed_sides * transaction_cost_pct

    # Net strategy return after costs
    df["strategy_return"] = df["strategy_return_gross"] - df["tx_cost"]

    # Equity curve
    df["equity"] = (1.0 + df["strategy_return"]).cumprod() * initial_capital

    # Metrics
    trades = int((pos_diff > 0).sum())
    total_return_pct = (df["equity"].iloc[-1] / initial_capital - 1.0) * 100.0
    max_equity = df["equity"].cummax()
    drawdown = df["equity"] / max_equity - 1.0
    max_drawdown_pct = drawdown.min() * 100.0

    rets = df["strategy_return"].dropna()
    mu = rets.mean()
    sigma = rets.std(ddof=0)
    minutes_per_year = 365 * 24 * 60
    if sigma == 0 or np.isnan(sigma):
        sharpe = np.nan
    else:
        sharpe = float((mu / sigma) * np.sqrt(minutes_per_year))
        
    metrics = {
        "final_equity": df["equity"].iloc[-1],
        "total_return_pct": total_return_pct,
        "max_drawdown_pct": max_drawdown_pct,
        "trades": trades,
        "sharpe": sharpe
    }
    if verbose:
        print("Simulation")
        print(f"Initial capital: ${initial_capital:,.2f}")
        print(f"Allow short enter: {allow_short_enter}")
        print(f"Transaction cost per side: {transaction_cost_pct*100:.4f}%")
        print("Results")
        print("------")
        print(f"Final equity: ${df['equity'].iloc[-1]:,.2f}")
        print(f"Total return: {total_return_pct:.2f}%")
        print(f"Max drawdown: {max_drawdown_pct:.2f}%")
        print(f"Number of trades: {trades}")
        print(f"Sharpe ratio: {sharpe}")

    return df, metrics


def compute_sma_signal(df, window):
    """Compute SMA and a price vs SMA signal on a copy, then return it."""
    out = df.copy()
    out["sma"] = out["mid_price"].rolling(window=window).mean()
    out["signal"] = 0
    out.loc[out["mid_price"] > out["sma"], "signal"] = 1
    out.loc[out["mid_price"] < out["sma"], "signal"] = -1
    return out

def run_sma_sweep(
    df,
    sma_periods=[5, 10, 20, 50],
    initial_capital=100_000.0,
    transaction_cost_pct=0.0002,
    allow_short_enter=False,
    backtest_func=None,
    price_col="mid_price",
    verbose=False,
):
    """
    Run multiple back tests over a set of SMA periods.
    Assumes `backtest_func` returns (df_result, metrics_dict).

    Returns
    -------
    metrics_df : pd.DataFrame
        One row per SMA period with final equity, total return, drawdown, Sharpe, and trades.
    results_by_period : dict[int, pd.DataFrame]
        Mapping from period to the full result dataframe of that run.
    equities : dict[str, pd.Series]
        Mapping from legend label to equity series, ready for plotting.
    """
    assert backtest_func is not None, "Please pass your backtest function via backtest_func"

    rows = []
    results_by_period = {}
    equities = {}

    for period in sma_periods:
        # period must be integer
        period = int(period)
        # prepare data for this period
        df_period = compute_sma_signal(df, window=period)

        # run back test
        df_res, metrics = backtest_func(
            df_period,
            sma_col="sma",
            price_col=price_col,
            signal_col="signal",
            initial_capital=initial_capital,
            allow_short_enter=allow_short_enter,
            transaction_cost_pct=transaction_cost_pct,
            verbose=verbose
        )

        # collect metrics with safe defaults
        final_equity = metrics.get("final_equity", float(df_res["equity"].iloc[-1]))
        total_return_pct = metrics.get("total_return_pct", None)
        max_drawdown_pct = metrics.get("max_drawdown_pct", None)
        sharpe = metrics.get("sharpe", None)
        num_trades = metrics.get("num_trades", metrics.get("trades", None))

        rows.append({
            "period": int(period),
            "final_equity": final_equity,
            "total_return_pct": total_return_pct,
            "max_drawdown_pct": max_drawdown_pct,
            "sharpe": sharpe,
            "num_trades": num_trades,
        })

        results_by_period[int(period)] = df_res
        equities[f"SMA {period}"] = df_res["equity"]

    metrics_df = pd.DataFrame(rows).set_index("period").sort_values("total_return_pct", ascending=False)
    return metrics_df, results_by_period, equities