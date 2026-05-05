"""
AI Trading Dashboard - POL/USDT (formerly MATIC)
Professional Streamlit app with real-time data, interactive Plotly charts,
ATR/ADX indicators, AI prediction, and risk management.
All-in-one file – fixed for Streamlit 1.57+.

FIXES vs original:
  1. ADX index alignment bug: pd.Series(arr, index=df.index) preserves datetime index
     so +DI / -DI / ADX are no longer all-NaN.
  2. load_data moved to module level so @st.cache_data actually works across reruns.
  3. fetch_binance_data wrapped in try/except with POL/USDT fallback (MATIC renamed Sep 2024).
  4. Volume color calculation vectorised (was slow iterrows loop).
  5. Removed redundant `position_size_pct = kelly_frac` alias.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import ccxt
from sklearn.linear_model import LinearRegression

# ------------------------ Page configuration ------------------------
st.set_page_config(
    page_title="POL/USDT AI Trading Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ------------------------ Helper functions ------------------------

# FIX 3: robust fetch with symbol fallback and error handling
@st.cache_data(ttl=60, show_spinner=False)
def fetch_binance_data(symbol: str = "POL/USDT", timeframe: str = "1h", limit: int = 500):
    """
    Fetch OHLCV data from Binance using ccxt.
    Tries the given symbol first; on failure, falls back from MATIC→POL or vice-versa.
    Returns DataFrame with columns: open, high, low, close, volume
    """
    exchange = ccxt.binance()
    fallback = {"POL/USDT": "MATIC/USDT", "MATIC/USDT": "POL/USDT"}.get(symbol)
    for attempt_symbol in ([symbol, fallback] if fallback else [symbol]):
        if attempt_symbol is None:
            continue
        try:
            ohlcv = exchange.fetch_ohlcv(attempt_symbol, timeframe, limit=limit)
            df = pd.DataFrame(
                ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"]
            )
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
            df.set_index("timestamp", inplace=True)
            return df
        except Exception:
            continue
    raise RuntimeError(
        f"Could not fetch data for {symbol} (or fallback). "
        "Check your internet connection and that the symbol is active on Binance."
    )


def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Average True Range (Wilder's smoothing)."""
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    true_range = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)
    return true_range.ewm(span=period, adjust=False).mean()


def calculate_adx(df: pd.DataFrame, period: int = 14):
    """
    Average Directional Index.

    FIX 1: pd.Series(arr, index=df.index) is used so that plus_dm_smoothed /
    minus_dm_smoothed share the same datetime index as atr.  Without this the
    division aligns on integer vs datetime index and produces all-NaN results.
    """
    high, low = df["high"], df["low"]

    up_move   = high.diff()
    down_move = -low.diff()

    plus_dm  = np.where((up_move > down_move) & (up_move  > 0), up_move,  0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)

    atr = calculate_atr(df, period)

    # --- FIX 1: preserve datetime index ---
    plus_dm_smoothed  = pd.Series(plus_dm,  index=df.index).ewm(span=period, adjust=False).mean()
    minus_dm_smoothed = pd.Series(minus_dm, index=df.index).ewm(span=period, adjust=False).mean()

    plus_di  = 100 * (plus_dm_smoothed  / atr)
    minus_di = 100 * (minus_dm_smoothed / atr)
    dx  = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-10)
    adx = dx.ewm(span=period, adjust=False).mean()
    return adx, plus_di, minus_di


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add ATR, ADX, and moving averages to dataframe."""
    df = df.copy()
    df["atr"] = calculate_atr(df)
    df["adx"], df["plus_di"], df["minus_di"] = calculate_adx(df)
    df["ma20"] = df["close"].rolling(window=20).mean()
    df["ma50"] = df["close"].rolling(window=50).mean()
    return df


def train_ai_model(df: pd.DataFrame, lookback: int = 10):
    """Train linear regression to predict next close price."""
    prices = df["close"].values
    X, y = [], []
    for i in range(lookback, len(prices) - 1):
        X.append(prices[i - lookback : i])
        y.append(prices[i + 1])
    if len(X) < 10:
        return None, 0
    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)
    mape = np.mean(np.abs(y - y_pred) / y) * 100
    confidence = max(0.0, min(100.0, 100.0 - mape))
    return model, confidence


def predict_next(model, df: pd.DataFrame, lookback: int = 10):
    """Predict next close price and direction."""
    last_prices = df["close"].iloc[-lookback:].values
    if len(last_prices) < lookback:
        return None, None
    pred_price = model.predict(last_prices.reshape(1, -1))[0]
    direction  = "UP" if pred_price > df["close"].iloc[-1] else "DOWN"
    return pred_price, direction


def generate_signal(df: pd.DataFrame, adx_threshold: float = 25, atr_vol_threshold: float = 0.05):
    """Generate trading signal based on ADX trend strength and MA crossover."""
    latest, prev = df.iloc[-1], df.iloc[-2] if len(df) > 1 else df.iloc[-1]
    atr_ratio = latest["atr"] / latest["close"]

    if atr_ratio >= atr_vol_threshold:
        return "HOLD", f"Volatility too high (ATR/price={atr_ratio:.3f})"
    if latest["adx"] < adx_threshold:
        return "HOLD", f"Trend weak (ADX={latest['adx']:.1f} < {adx_threshold})"
    if latest["ma20"] > latest["ma50"] and prev["ma20"] <= prev["ma50"]:
        return "BUY",  f"MA20 crossed above MA50, ADX={latest['adx']:.1f}"
    if latest["ma20"] < latest["ma50"] and prev["ma20"] >= prev["ma50"]:
        return "SELL", f"MA20 crossed below MA50, ADX={latest['adx']:.1f}"
    return "HOLD", "No MA crossover"


def ai_confirm_signal(base_signal: str, ai_direction: str, confidence: float):
    """Confirm or override signal based on AI prediction."""
    if confidence < 60:
        return base_signal, "low confidence"
    if base_signal == "BUY"  and ai_direction == "UP":
        return "BUY",  "AI confirmed uptrend"
    if base_signal == "SELL" and ai_direction == "DOWN":
        return "SELL", "AI confirmed downtrend"
    return "HOLD", "AI direction mismatch"


def kelly_criterion(win_rate: float, avg_win_loss_ratio: float) -> float:
    """Kelly fraction for position sizing (capped at 25 %)."""
    if avg_win_loss_ratio <= 0:
        return 0.0
    b, p = avg_win_loss_ratio, win_rate
    kelly = (p * b - (1 - p)) / b
    return max(0.0, min(kelly, 0.25))


def plot_candlestick_with_signals(df: pd.DataFrame, buy_signals: list, sell_signals: list):
    """Create interactive Plotly chart with candlesticks, volume, MA, and signals."""
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.7, 0.3],
    )

    fig.add_trace(
        go.Candlestick(
            x=df.index, open=df["open"], high=df["high"], low=df["low"], close=df["close"],
            name="Price",
        ),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(x=df.index, y=df["ma20"], line=dict(color="orange", width=1), name="MA20"),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(x=df.index, y=df["ma50"], line=dict(color="blue", width=1), name="MA50"),
        row=1, col=1,
    )

    if buy_signals:
        buy_df = df.loc[buy_signals]
        fig.add_trace(
            go.Scatter(
                x=buy_df.index, y=buy_df["close"], mode="markers",
                marker=dict(symbol="triangle-up", size=12, color="green"),
                name="Buy Signal",
            ),
            row=1, col=1,
        )
    if sell_signals:
        sell_df = df.loc[sell_signals]
        fig.add_trace(
            go.Scatter(
                x=sell_df.index, y=sell_df["close"], mode="markers",
                marker=dict(symbol="triangle-down", size=12, color="red"),
                name="Sell Signal",
            ),
            row=1, col=1,
        )

    # FIX 5: vectorised volume colours (was a slow iterrows loop)
    vol_colors = np.where(df["close"] >= df["open"], "green", "red")
    fig.add_trace(
        go.Bar(x=df.index, y=df["volume"], name="Volume", marker_color=vol_colors),
        row=2, col=1,
    )

    fig.update_layout(
        title="POL/USDT - Price Chart",
        xaxis_title="Time",
        yaxis_title="Price (USDT)",
        height=700,
        hovermode="x unified",
        template="plotly_dark",
    )
    fig.update_yaxes(title_text="Price",  row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1)
    return fig


# FIX 2: load_data at module level so @st.cache_data persists across reruns
@st.cache_data(ttl=60, show_spinner=False)
def load_data(timeframe: str) -> pd.DataFrame:
    return fetch_binance_data("POL/USDT", timeframe, limit=300)


# ------------------------ Main App ------------------------
def main():
    st.title("🤖 AI Trading Dashboard")
    st.markdown("### POL/USDT (formerly MATIC) | Real-time signals with ATR, ADX & Linear Regression")

    # Sidebar
    st.sidebar.header("Controls")
    timeframe = st.sidebar.selectbox(
        "Timeframe",
        options=["1m", "5m", "15m", "1h", "4h", "1d"],
        index=3,
    )
    if st.sidebar.button("🔄 Refresh Data", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

    st.sidebar.markdown("---")
    st.sidebar.info(
        "**Strategy:**\n"
        "- ADX > 25 for trend strength\n"
        "- MA20/MA50 crossover\n"
        "- Volatility filter (ATR/price < 5 %)\n"
        "- AI confirmation (Linear Regression)\n"
        "- Risk: Kelly Criterion + ATR stop-loss"
    )

    # Load data
    with st.spinner("Fetching market data…"):
        try:
            df_raw = load_data(timeframe)
        except RuntimeError as e:
            st.error(str(e))
            st.stop()

    df = add_indicators(df_raw)

    current_price = df["close"].iloc[-1]
    current_atr   = df["atr"].iloc[-1]
    current_adx   = df["adx"].iloc[-1]
    atr_ratio      = current_atr / current_price

    base_signal, signal_reason = generate_signal(df)

    model, ai_conf = train_ai_model(df)
    if model:
        pred_price, ai_dir = predict_next(model, df)
        if pred_price:
            confidence   = ai_conf
            final_signal, confirm_msg = ai_confirm_signal(base_signal, ai_dir, confidence)
        else:
            pred_price, ai_dir, confidence = current_price, "UNKNOWN", 0.0
            final_signal, confirm_msg = base_signal, "AI unavailable"
    else:
        pred_price, ai_dir, confidence = current_price, "UNKNOWN", 0.0
        final_signal, confirm_msg = base_signal, "AI not trained (insufficient data)"

    # Risk management
    win_rate, avg_win_loss = 0.55, 1.4
    kelly_frac      = kelly_criterion(win_rate, avg_win_loss)   # FIX 6: removed redundant alias
    stop_loss_price = current_price - 2 * current_atr

    # Metrics row
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Current Price", f"${current_price:.4f}")
    with col2:
        color = {"BUY": "green", "SELL": "red"}.get(final_signal, "gray")
        st.markdown(f"<h3 style='text-align:center;color:{color}'>{final_signal}</h3>",
                    unsafe_allow_html=True)
        st.caption(f"Base: {base_signal} | {confirm_msg}")
    with col3:
        delta_pct = ((pred_price / current_price) - 1) * 100
        st.metric(
            "AI Prediction", f"${pred_price:.4f}",
            delta=f"{delta_pct:.2f}%",
            delta_color="normal" if ai_dir == "UP" else "inverse",
        )
        st.caption(f"Direction: {ai_dir} | Confidence: {confidence:.1f}%")
    with col4:
        st.metric("Trend Strength (ADX)", f"{current_adx:.1f}",
                  help=">25 indicates a trending market")

    # Main chart
    adx_threshold = 25
    buy_signals, sell_signals = [], []
    for i in range(50, len(df)):
        if df["adx"].iloc[i] > adx_threshold:
            if df["ma20"].iloc[i] > df["ma50"].iloc[i] and df["ma20"].iloc[i - 1] <= df["ma50"].iloc[i - 1]:
                buy_signals.append(df.index[i])
            elif df["ma20"].iloc[i] < df["ma50"].iloc[i] and df["ma20"].iloc[i - 1] >= df["ma50"].iloc[i - 1]:
                sell_signals.append(df.index[i])

    fig = plot_candlestick_with_signals(df, buy_signals, sell_signals)
    st.plotly_chart(fig, use_container_width=True)

    # Indicator panels
    st.markdown("---")
    col5, col6 = st.columns(2)
    with col5:
        st.subheader("📊 ATR")
        fig_atr = go.Figure()
        fig_atr.add_trace(go.Scatter(x=df.index, y=df["atr"], mode="lines",
                                     name="ATR", line=dict(color="orange")))
        fig_atr.update_layout(title="Average True Range (ATR)", height=250, template="plotly_dark")
        st.plotly_chart(fig_atr, use_container_width=True)
    with col6:
        st.subheader("📈 ADX & DI")
        fig_adx = go.Figure()
        fig_adx.add_trace(go.Scatter(x=df.index, y=df["adx"],      mode="lines", name="ADX",  line=dict(color="blue")))
        fig_adx.add_trace(go.Scatter(x=df.index, y=df["plus_di"],  mode="lines", name="+DI",  line=dict(color="green")))
        fig_adx.add_trace(go.Scatter(x=df.index, y=df["minus_di"], mode="lines", name="-DI",  line=dict(color="red")))
        fig_adx.add_hline(y=25, line_dash="dash", line_color="gray", annotation_text="Threshold")
        fig_adx.update_layout(title="Directional Movement Index", height=250, template="plotly_dark")
        st.plotly_chart(fig_adx, use_container_width=True)

    # Risk panel
    st.markdown("---")
    st.subheader("💰 Risk Management")
    col7, col8, col9 = st.columns(3)
    with col7:
        st.metric("Kelly Fraction (risk per trade)", f"{kelly_frac:.2%}")
        st.caption(f"Based on win rate {win_rate:.0%}, avg win/loss {avg_win_loss:.1f}")
    with col8:
        st.metric("Position Size (recommended)", f"{kelly_frac:.2%} of capital")
        st.caption("Adjust according to your risk tolerance")
    with col9:
        st.metric("ATR Stop-Loss (2×)", f"${stop_loss_price:.4f}")
        st.caption(f"Distance: {2 * current_atr:.4f} USDT ({2 * current_atr / current_price:.2%})")

    # Explanation expander
    st.markdown("---")
    with st.expander("🔍 Why this signal? (Strategy details)"):
        trend_label = "Trending" if current_adx > 25 else "Range-bound"
        ma_label    = "MA20 above MA50" if df["ma20"].iloc[-1] > df["ma50"].iloc[-1] else "MA20 below MA50"
        vol_label   = "Acceptable" if atr_ratio < 0.05 else "Too high"
        st.write(f"""
**Base Strategy:**
- ADX = {current_adx:.1f} → {trend_label}
- MA20 (={df["ma20"].iloc[-1]:.4f}) vs MA50 (={df["ma50"].iloc[-1]:.4f}) → {ma_label}
- Volatility: ATR/Price = {atr_ratio:.2%} → {vol_label}
- Base signal: {base_signal} → {signal_reason}

**AI Confirmation:**
- Predicted next price: ${pred_price:.4f} ({ai_dir})
- Confidence: {confidence:.1f}%
- Final signal: {final_signal} because {confirm_msg}

**Risk:**
- Kelly suggests risking {kelly_frac:.1%} of capital per trade
- Stop-loss at ${stop_loss_price:.4f} (2× ATR)
        """)

    st.markdown("---")
    st.caption(
        "Data from Binance | AI model: Linear Regression on closing prices | "
        "Dashboard refreshes every 60 s (cached)"
    )


if __name__ == "__main__":
    main()
