import numpy as np
import pandas as pd

def backtest_strategy(
    df,
    sma_col="sma_10",
    price_col="mid_price",
    signal_col="signal",
    initial_capital=100_000.0,
    allow_short_enter=False,
    transaction_cost_pct=0.0002  # 0.02% per side
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

    return df
