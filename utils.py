import matplotlib.pyplot as plt
import pandas as pd

def plot_strategy_overview(
    df,
    n_points=200,
    price_col="mid_price",
    sma_col="sma",
    signal_col="signal",
    position_col="position",
    equity_col="equity",
    title=None,
    shade_zones=True
):
    """
    Plot price, SMA, entry signals, held position, and equity on twin axes.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame that already contains price, SMA, signal, position, and equity.
    n_points : int
        Number of initial rows to display.
    price_col, sma_col, signal_col, position_col, equity_col : str
        Column names for each quantity.
    title : str or None
        Custom title. If None a default title is used.
    shade_zones : bool
        If True, shades long and short holding periods.
    """

    if n_points is None or n_points <= 0:
        subset = df
        n_shown = len(df)
    else:
        subset = df.iloc[:min(n_points, len(df))]
        n_shown = len(subset)

    fig, ax1 = plt.subplots(figsize=(14, 6))

    # Primary axis: price and SMA
    ax1.plot(subset.index, subset[price_col], label="Mid Price", color="blue", linewidth=1.2)
    ax1.plot(subset.index, subset[sma_col], label="10 period SMA", color="orange", linewidth=1.5)

    # Signals
    long_signals = subset[subset[signal_col] == 1]
    short_signals = subset[subset[signal_col] == -1]

    ax1.scatter(long_signals.index, long_signals[price_col],
                label="Long Signal", color="green", marker="^", s=60, zorder=5)
    ax1.scatter(short_signals.index, short_signals[price_col],
                label="Short Signal", color="red", marker="v", s=60, zorder=5)

    # Shaded position zones
    if shade_zones:
        ymin = subset[price_col].min()
        ymax = subset[price_col].max()
        ax1.fill_between(
            subset.index, ymin, ymax,
            where=subset[position_col] > 0,
            color="green", alpha=0.08, label="Long Position Zone"
        )
        ax1.fill_between(
            subset.index, ymin, ymax,
            where=subset[position_col] < 0,
            color="red", alpha=0.08, label="Short Position Zone"
        )

    # Secondary axis: equity
    ax2 = ax1.twinx()
    ax2.plot(subset.index, subset[equity_col], label="Equity USD", color="black", linewidth=1.8, linestyle="--")

    # Labels and title
    ax1.set_xlabel("Time")
    ax1.set_ylabel("Price")
    ax2.set_ylabel("Equity USD")

    if title is None:
        title = f"Trading Strategy Overview first {n_shown} points"
    ax1.set_title(title)
    ax1.grid(True)

    # Unified legend
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc="upper left")

    plt.tight_layout()
    plt.show()
    return fig, ax1, ax2


def plot_strategy_overview_multi(
    df: pd.DataFrame,
    equities: dict,
    n_points: int = 200,
    price_col: str = "mid_price",
    sma_col: str = "sma",
    signal_col: str = "signal",
    position_col: str = "position",
    title: str = None,
    shade_zones: bool = True
):
    """
    Plot price, SMA, signals, position shading on the left axis
    and multiple equity curves on the right axis.

    Parameters
    ----------
    df : DataFrame with price, SMA, signal, position, and possibly equity columns.
    equities : dict
        Mapping from label -> equity source
        Each value can be:
          - a column name in df
          - a pandas Series indexed like df
    n_points : number of initial rows to display. Use None or <= 0 to show all.
    """

    # Select subset
    if n_points is None or n_points <= 0:
        subset = df
        n_shown = len(df)
    else:
        subset = df.iloc[:min(n_points, len(df))]
        n_shown = len(subset)

    fig, ax1 = plt.subplots(figsize=(14, 6))

    # Primary axis: price and SMA
    ax1.plot(subset.index, subset[price_col], label="Mid Price", color="blue", linewidth=1.2)
    ax1.plot(subset.index, subset[sma_col], label="10 period SMA", color="orange", linewidth=1.5)

    # Signals
    long_signals = subset[subset[signal_col] == 1]
    short_signals = subset[subset[signal_col] == -1]

    ax1.scatter(long_signals.index, long_signals[price_col],
                label="Long Signal", color="green", marker="^", s=60, zorder=5)
    ax1.scatter(short_signals.index, short_signals[price_col],
                label="Short Signal", color="red", marker="v", s=60, zorder=5)

    # Position shading
    if shade_zones:
        ymin = subset[price_col].min()
        ymax = subset[price_col].max()
        ax1.fill_between(subset.index, ymin, ymax, where=subset[position_col] > 0, alpha=0.08, label="Long Position Zone")
        ax1.fill_between(subset.index, ymin, ymax, where=subset[position_col] < 0, alpha=0.08, label="Short Position Zone")

    # Secondary axis for multiple equity curves
    ax2 = ax1.twinx()

    for label, src in equities.items():
        if isinstance(src, str):
            series = df[src]
        else:
            series = pd.Series(src)
        # align to subset index
        series = series.reindex(df.index)
        series_sub = series.loc[subset.index]
        ax2.plot(series_sub.index, series_sub.values, label=label, linewidth=1.8, linestyle="--")

    # Labels and title
    ax1.set_xlabel("Time")
    ax1.set_ylabel("Price")
    ax2.set_ylabel("Equity USD")
    if title is None:
        title = f"Trading Strategy Overview first {n_shown} points"
    ax1.set_title(title)
    ax1.grid(True)

    # Unified legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

    plt.tight_layout()
    plt.show()
    return fig, ax1, ax2


def plot_equity_curves(
    equities: dict,
    n_points: int = None,
    title: str = "Equity Curve Comparison",
    start_capital: float = 100_000.0
):
    """
    Plot multiple equity curves (e.g., from different SMA strategies).

    Parameters
    ----------
    equities : dict[str, pd.Series]
        Mapping of label -> equity series (must have aligned datetime index).
    n_points : int or None
        Number of points from the start to plot. If None, show all.
    title : str
        Plot title.
    start_capital : float
        Optional reference line for starting equity.
    """

    if not equities:
        raise ValueError("No equity curves provided")

    # Choose an index for consistent plotting (use the first series)
    ref_series = next(iter(equities.values()))
    if n_points:
        ref_series = ref_series.iloc[:n_points]
        x_index = ref_series.index
    else:
        x_index = ref_series.index

    fig, ax = plt.subplots(figsize=(12, 6))

    for label, series in equities.items():
        s = series.reindex(x_index)
        if n_points:
            s = s.iloc[:n_points]
        ax.plot(s.index, s.values, label=label, linewidth=1.8)

    # Reference line for starting capital
    ax.axhline(start_capital, linestyle="--", color="gray", linewidth=1, label="Initial capital")

    ax.set_title(title)
    ax.set_xlabel("Time")
    ax.set_ylabel("Equity (USD)")
    ax.legend(loc="upper left")
    ax.grid(True)
    plt.tight_layout()
    plt.show()

    return fig, ax
