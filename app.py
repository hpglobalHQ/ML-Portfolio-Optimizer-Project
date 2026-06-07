# app.py

import streamlit as st
import pandas as pd
import numpy as np

from data_pipeline import (
    load_cross_sectional_data
)

from model import (
    train_model,
    FEATURES
)

from validation import (
    walk_forward_validation
)


# -------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------
st.set_page_config(

    page_title='MARKET SENSE AI',

    layout='wide'
)


# -------------------------------------------------
# LIGHT THEME — COSMETIC ONLY (no logic changes)
# -------------------------------------------------
st.markdown(
    """
    <style>
    /* ── Google Fonts ── */
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=DM+Mono:wght@400;500&display=swap');

    /* ── Root palette ── */
    :root {
        --bg:          #F7F8FA;
        --surface:     #FFFFFF;
        --surface-alt: #EEF1F6;
        --border:      #DDE2EC;
        --accent:      #2563EB;
        --accent-soft: #DBEAFE;
        --text-head:   #0F172A;
        --text-body:   #334155;
        --text-muted:  #64748B;
        --green:       #16A34A;
        --amber:       #D97706;
        --red:         #DC2626;
        --radius:      10px;
        --shadow:      0 1px 4px rgba(0,0,0,.06), 0 4px 16px rgba(0,0,0,.04);
    }

    /* ── Global reset ── */
    html, body, [data-testid="stAppViewContainer"],
    [data-testid="stMain"], .main {
        background-color: var(--bg) !important;
        font-family: 'DM Sans', sans-serif !important;
        color: var(--text-body) !important;
    }

    /* ── Sidebar ── */
    [data-testid="stSidebar"] {
        background: var(--surface) !important;
        border-right: 1px solid var(--border) !important;
        box-shadow: 2px 0 12px rgba(0,0,0,.04) !important;
    }
    [data-testid="stSidebar"] * {
        font-family: 'DM Sans', sans-serif !important;
        color: var(--text-body) !important;
    }
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {
        color: var(--text-head) !important;
        font-weight: 600 !important;
    }

    /* ── Main title ── */
    h1 {
        font-family: 'DM Sans', sans-serif !important;
        font-weight: 700 !important;
        font-size: 2rem !important;
        color: var(--text-head) !important;
        letter-spacing: -0.5px !important;
        padding-bottom: 4px !important;
        border-bottom: 2px solid var(--accent) !important;
        margin-bottom: 1rem !important;
    }

    /* ── Section headers ── */
    h2, h3 {
        font-family: 'DM Sans', sans-serif !important;
        font-weight: 600 !important;
        color: var(--text-head) !important;
        letter-spacing: -0.3px !important;
        margin-top: 1.6rem !important;
    }

    /* ── Metric cards ── */
    [data-testid="stMetric"] {
        background: var(--surface) !important;
        border: 1px solid var(--border) !important;
        border-radius: var(--radius) !important;
        padding: 14px 18px !important;
        box-shadow: var(--shadow) !important;
        transition: box-shadow .2s ease !important;
    }
    [data-testid="stMetric"]:hover {
        box-shadow: 0 4px 20px rgba(37,99,235,.10) !important;
        border-color: var(--accent) !important;
    }
    [data-testid="stMetricLabel"] {
        font-size: 0.72rem !important;
        font-weight: 600 !important;
        text-transform: uppercase !important;
        letter-spacing: .6px !important;
        color: var(--text-muted) !important;
    }
    [data-testid="stMetricValue"] {
        font-size: 1.5rem !important;
        font-weight: 700 !important;
        color: var(--text-head) !important;
        font-family: 'DM Mono', monospace !important;
    }

    /* ── DataFrames ── */
    [data-testid="stDataFrame"],
    [data-testid="stDataFrameGlideDataEditor"] {
        border: 1px solid var(--border) !important;
        border-radius: var(--radius) !important;
        overflow: hidden !important;
        background: var(--surface) !important;
        box-shadow: var(--shadow) !important;
    }

    /* ── Spinners & alerts ── */
    [data-testid="stAlert"] {
        border-radius: var(--radius) !important;
        font-family: 'DM Sans', sans-serif !important;
        font-size: 0.875rem !important;
    }
    div[data-testid="stAlert"][kind="success"],
    div.stSuccess {
        background: #F0FDF4 !important;
        border-left: 4px solid var(--green) !important;
        color: #15803D !important;
    }
    div[data-testid="stAlert"][kind="info"],
    div.stInfo {
        background: var(--accent-soft) !important;
        border-left: 4px solid var(--accent) !important;
        color: #1E40AF !important;
    }

    /* ── Sliders ── */
    [data-testid="stSlider"] > div > div > div {
        background: var(--accent) !important;
    }
    [data-testid="stSlider"] [role="slider"] {
        background: var(--accent) !important;
        border: 2px solid white !important;
        box-shadow: 0 0 0 3px var(--accent-soft) !important;
    }

    /* ── Line chart container ── */
    [data-testid="stArrowVegaLiteChart"],
    [data-testid="stVegaLiteChart"] {
        background: var(--surface) !important;
        border: 1px solid var(--border) !important;
        border-radius: var(--radius) !important;
        padding: 12px !important;
        box-shadow: var(--shadow) !important;
    }

    /* ── JSON viewer ── */
    [data-testid="stJson"] {
        background: var(--surface-alt) !important;
        border: 1px solid var(--border) !important;
        border-radius: var(--radius) !important;
        font-family: 'DM Mono', monospace !important;
        font-size: 0.8rem !important;
    }

    /* ── Dividers ── */
    hr {
        border-color: var(--border) !important;
    }

    /* ── Paragraph / markdown body ── */
    p, li, .stMarkdown {
        font-size: 0.9rem !important;
        line-height: 1.65 !important;
        color: var(--text-body) !important;
    }
    ul li {
        margin-bottom: 4px !important;
    }

    /* ── Top toolbar strip ── */
    [data-testid="stToolbar"] {
        background: var(--surface) !important;
        border-bottom: 1px solid var(--border) !important;
    }

    /* ── Spinner text ── */
    .stSpinner > div {
        border-top-color: var(--accent) !important;
    }

    /* ── Column gaps ── */
    [data-testid="column"] {
        padding: 4px 6px !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# -------------------------------------------------
# TITLE  — enhanced hero section (cosmetic only)
# -------------------------------------------------
st.markdown(
    """
    <style>
    /* ── Hero block ── */
    .hero-wrap {
        background: linear-gradient(135deg, #EFF6FF 0%, #F8FAFF 60%, #F0FDF4 100%);
        border: 1px solid #C7D7F4;
        border-radius: 16px;
        padding: 36px 40px 32px 40px;
        margin-bottom: 28px;
        position: relative;
        overflow: hidden;
    }
    .hero-wrap::before {
        content: '';
        position: absolute;
        top: -60px; right: -60px;
        width: 220px; height: 220px;
        background: radial-gradient(circle, rgba(37,99,235,0.08) 0%, transparent 70%);
        border-radius: 50%;
        pointer-events: none;
    }
    .hero-badge {
        display: inline-flex;
        align-items: center;
        gap: 5px;
        background: #DBEAFE;
        color: #1D4ED8;
        font-size: 0.68rem;
        font-weight: 700;
        letter-spacing: .8px;
        text-transform: uppercase;
        padding: 4px 12px;
        border-radius: 20px;
        border: 1px solid #BFDBFE;
        margin-bottom: 14px;
    }
    .hero-title {
        font-family: 'DM Sans', sans-serif;
        font-size: 2.4rem;
        font-weight: 800;
        color: #0F172A;
        letter-spacing: -1px;
        line-height: 1.15;
        margin: 0 0 10px 0;
    }
    .hero-title span {
        color: #2563EB;
    }
    .hero-sub {
        font-family: 'DM Sans', sans-serif;
        font-size: 0.95rem;
        color: #475569;
        line-height: 1.6;
        max-width: 560px;
        margin-bottom: 28px;
    }
    /* ── Arch cards grid ── */
    .arch-grid {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 10px;
    }
    .arch-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 12px 14px;
        display: flex;
        align-items: flex-start;
        gap: 10px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        transition: border-color .2s, box-shadow .2s;
    }
    .arch-card:hover {
        border-color: #93C5FD;
        box-shadow: 0 3px 12px rgba(37,99,235,0.08);
    }
    .arch-icon {
        font-size: 1.1rem;
        flex-shrink: 0;
        margin-top: 1px;
    }
    .arch-text {
        font-family: 'DM Sans', sans-serif;
        font-size: 0.82rem;
        font-weight: 500;
        color: #1E293B;
        line-height: 1.3;
    }
    /* ── Divider after hero ── */
    .hero-divider {
        height: 1px;
        background: linear-gradient(90deg, #2563EB22, #2563EB55 30%, #2563EB22);
        margin: 8px 0 28px 0;
        border: none;
    }
    </style>

    <div class="hero-wrap">
        <div class="hero-badge">📊 &nbsp;Educational &amp; Research Use Only</div>
        <div class="hero-title">Market<span>Sense</span> AI</div>
        <div class="hero-sub">
            Machine-learning powered cross-sectional quant research system.
        </div>
        <div class="arch-grid">
            <div class="arch-card"><span class="arch-icon">🔀</span><span class="arch-text">Cross-sectional stock ranking</span></div>
            <div class="arch-card"><span class="arch-icon">⚖️</span><span class="arch-text">Market-neutral alpha strategy</span></div>
            <div class="arch-card"><span class="arch-icon">📉</span><span class="arch-text">Long-short portfolio construction</span></div>
            <div class="arch-card"><span class="arch-icon">🛡️</span><span class="arch-text">Volatility-aware risk management</span></div>
            <div class="arch-card"><span class="arch-icon">🚶</span><span class="arch-text">Walk-forward backtesting</span></div>
            <div class="arch-card"><span class="arch-icon">🔒</span><span class="arch-text">Leakage-safe validation</span></div>
            <div class="arch-card"><span class="arch-icon">🧬</span><span class="arch-text">Multi-factor feature engineering</span></div>
            <div class="arch-card"><span class="arch-icon">📐</span><span class="arch-text">IC, ICIR &amp; turnover analytics</span></div>
            <div class="arch-card"><span class="arch-icon">🔬</span><span class="arch-text">Signal-quality diagnostics</span></div>
        </div>
    </div>
    <hr class="hero-divider" />
    """,
    unsafe_allow_html=True
)


# -------------------------------------------------
# SIDEBAR
# -------------------------------------------------
st.sidebar.header(
    '⚙️ Portfolio Settings'
)


# -------------------------------------------------
# TOP N
# -------------------------------------------------
top_n = st.sidebar.slider(

    'Top Stocks',

    min_value=1,

    max_value=10,

    value=5
)


# -------------------------------------------------
# CONFIDENCE THRESHOLD
# -------------------------------------------------
confidence_threshold = st.sidebar.slider(

    'Signal Confidence Threshold',

    min_value=0.5,

    max_value=3.0,

    value=1.0,

    step=0.1
)


# -------------------------------------------------
# REBALANCE FREQUENCY
# -------------------------------------------------
rebalance_frequency = st.sidebar.slider(

    'Rebalance Frequency (Days)',

    min_value=5,

    max_value=60,

    value=20
)


# -------------------------------------------------
# LOAD DATA
# -------------------------------------------------
with st.spinner(
    'Loading cross-sectional dataset...'
):

    full_df, market_df = (
        load_cross_sectional_data()
    )

st.success(
    'Cross-sectional dataset loaded.'
)


# -------------------------------------------------
# DATASET OVERVIEW
# -------------------------------------------------
st.subheader(
    '📊 Dataset Overview'
)

col1, col2, col3 = st.columns(3)

with col1:

    st.metric(
        'Total Rows',
        len(full_df)
    )

with col2:

    st.metric(
        'Stocks',
        full_df['Ticker'].nunique()
    )

with col3:

    st.metric(
        'Features',
        len(FEATURES)
    )


# -------------------------------------------------
# DATE RANGE
# -------------------------------------------------
st.subheader(
    '📅 Dataset Date Range'
)

col1, col2 = st.columns(2)

with col1:

    st.metric(
        'Start Date',
        str(
            full_df.index.min().date()
        )
    )

with col2:

    st.metric(
        'End Date',
        str(
            full_df.index.max().date()
        )
    )


# -------------------------------------------------
# STOCK UNIVERSE
# -------------------------------------------------
st.subheader(
    '📦 Stock Universe'
)

universe_list = sorted(
    list(
        full_df['Ticker']
        .unique()
    )
)

st.json(
    universe_list
)


# -------------------------------------------------
# SECTOR BREAKDOWN
# -------------------------------------------------
st.subheader(
    '🏢 Sector Breakdown'
)

sector_counts = (

    full_df[
        ['Ticker', 'Sector']
    ]

    .drop_duplicates()

    ['Sector']

    .value_counts()
)

sector_df = pd.DataFrame({

    'Sector': sector_counts.index,

    'Stocks': sector_counts.values
})

st.dataframe(
    sector_df,
    width='stretch'
)


# -------------------------------------------------
# PREPARE DATA
# -------------------------------------------------
X = full_df[FEATURES]

y = full_df['Target']


# -------------------------------------------------
# FEATURE NULL CHECK
# -------------------------------------------------
st.subheader(
    '🧪 Feature Integrity Check'
)

null_counts = X.isnull().sum()

null_df = pd.DataFrame({

    'Feature': null_counts.index,

    'Null Count': null_counts.values
})

st.dataframe(
    null_df,
    width='stretch'
)


# -------------------------------------------------
# TRAIN MODEL
# -------------------------------------------------
with st.spinner(
    'Training institutional ranking model...'
):

    trained_model = train_model(

        X,

        y,

        debug=True
    )

st.success(
    'Model training complete.'
)


# -------------------------------------------------
# MODEL DIAGNOSTICS
# -------------------------------------------------
st.subheader(
    '🧠 Model Diagnostics'
)

col1, col2, col3, col4 = st.columns(4)

with col1:

    st.metric(
        'Validation MSE',
        round(
            trained_model['mse'],
            6
        )
    )

with col2:

    st.metric(
        'Validation MAE',
        round(
            trained_model['mae'],
            6
        )
    )

with col3:

    st.metric(
        'Validation IC',
        round(
            trained_model['ic'],
            6
        )
    )

with col4:

    st.metric(
        'Average Daily IC',
        round(
            trained_model[
                'avg_daily_ic'
            ],
            6
        )
    )


# -------------------------------------------------
# FEATURE IMPORTANCE
# -------------------------------------------------
st.subheader(
    '📈 Feature Importance'
)

feature_importance = (
    trained_model[
        'feature_importance'
    ]
)

if feature_importance is not None:

    importance_df = pd.DataFrame({

        'Feature': FEATURES,

        'Importance': feature_importance
    })

    importance_df = (

        importance_df
        .sort_values(
            'Importance',
            ascending=False
        )
    )

    st.dataframe(
        importance_df,
        width='stretch'
    )


# -------------------------------------------------
# FEATURE CORRELATION
# -------------------------------------------------
st.subheader(
    '🔗 Feature Correlation Matrix'
)

correlation_df = (

    full_df[FEATURES]
    .corr()
)

st.dataframe(
    correlation_df,
    width='stretch'
)


# -------------------------------------------------
# FEATURE SUMMARY
# -------------------------------------------------
st.subheader(
    '📊 Feature Summary Statistics'
)

feature_summary = (

    full_df[FEATURES]
    .describe()
    .T
)

st.dataframe(
    feature_summary,
    width='stretch'
)


# -------------------------------------------------
# TARGET DISTRIBUTION
# -------------------------------------------------
st.subheader(
    '🎯 Target Distribution'
)

target_summary = pd.DataFrame({

    'Statistic': [

        'Mean',
        'Std',
        'Min',
        'Max',
        'Median'
    ],

    'Value': [

        full_df['Target'].mean(),
        full_df['Target'].std(),
        full_df['Target'].min(),
        full_df['Target'].max(),
        full_df['Target'].median()
    ]
})

st.dataframe(
    target_summary,
    width='stretch'
)


# -------------------------------------------------
# VALIDATION
# -------------------------------------------------
with st.spinner(
    'Running institutional walk-forward backtest...'
):

    results = walk_forward_validation(

        full_df,

        market_df,

        trained_model,

        top_n=top_n,

        confidence_threshold=confidence_threshold,

        rebalance_frequency=rebalance_frequency,

        debug=True
    )

st.success(
    'Backtest complete.'
)


# -------------------------------------------------
# PERFORMANCE METRICS
# -------------------------------------------------
st.subheader(
    '💰 Portfolio Metrics'
)

col1, col2, col3 = st.columns(3)

with col1:

    st.metric(

        'Total Return',

        f"{results['total_return'] * 100:.2f}%"
    )

with col2:

    st.metric(

        'Annualized Return',

        f"{results['annualized_return'] * 100:.2f}%"
    )

with col3:

    st.metric(

        'Sharpe Ratio',

        round(
            results['sharpe_ratio'],
            2
        )
    )


col4, col5, col6 = st.columns(3)

with col4:

    st.metric(

        'Sortino Ratio',

        round(
            results['sortino_ratio'],
            2
        )
    )

with col5:

    st.metric(

        'Win Rate',

        f"{results['win_rate'] * 100:.2f}%"
    )

with col6:

    st.metric(

        'Average Return',

        round(
            results['avg_return'],
            4
        )
    )


col7, col8, col9 = st.columns(3)

with col7:

    st.metric(

        'Max Drawdown',

        f"{results['max_drawdown'] * 100:.2f}%"
    )

with col8:

    st.metric(

        'Total Trades',

        results['total_trades']
    )

with col9:

    st.metric(

        'SPY Benchmark',

        f"{results['benchmark_return'] * 100:.2f}%"
    )


# -------------------------------------------------
# ALPHA METRICS
# -------------------------------------------------
st.subheader(
    '📊 Alpha vs Benchmark'
)

col1, col2, col3 = st.columns(3)

with col1:

    st.metric(

        'Excess Return',

        f"{results['excess_return'] * 100:.2f}%"
    )

with col2:

    st.metric(

        'Average Daily IC',

        round(
            results['avg_daily_ic'],
            6
        )
    )

with col3:

    st.metric(

        'ICIR',

        round(
            results['icir'],
            4
        )
    )


# -------------------------------------------------
# DISTRIBUTION METRICS
# -------------------------------------------------
st.subheader(
    '📊 Distribution Metrics'
)

col1, col2, col3 = st.columns(3)

with col1:

    st.metric(

        'Median Return',

        round(
            results['median_return'],
            4
        )
    )

with col2:

    st.metric(

        'Top 10%',

        round(
            results[
                'top_10_percentile'
            ],
            4
        )
    )

with col3:

    st.metric(

        'Bottom 10%',

        round(
            results[
                'bottom_10_percentile'
            ],
            4
        )
    )


# -------------------------------------------------
# SIGNAL QUALITY
# -------------------------------------------------
st.subheader(
    '🧪 Alpha Diagnostics'
)

col1, col2, col3, col4 = st.columns(4)

with col1:

    st.metric(

        'Prediction Std',

        round(
            results[
                'prediction_std'
            ],
            6
        )
    )

with col2:

    st.metric(

        'Prediction Mean',

        round(
            results[
                'prediction_mean'
            ],
            6
        )
    )

with col3:

    st.metric(

        'Signal Rate',

        round(
            results[
                'signal_rate'
            ],
            4
        )
    )

with col4:

    st.metric(

        'Portfolio Win Rate',

        round(
            results[
                'portfolio_win_rate'
            ] * 100,
            2
        )
    )


# -------------------------------------------------
# TURNOVER ANALYTICS
# -------------------------------------------------
st.subheader(
    '🔄 Turnover Analytics'
)

col1, col2 = st.columns(2)

with col1:

    st.metric(
        'Average Turnover',
        round(
            results['avg_turnover'],
            4
        )
    )

with col2:

    st.metric(
        'Rebalance Frequency',
        rebalance_frequency
    )


# -------------------------------------------------
# EQUITY CURVE
# -------------------------------------------------
st.subheader(
    '📈 Equity Curve'
)

equity_df = pd.DataFrame({

    'Date': results['equity_dates'],

    'Portfolio Equity': results['equity_curve']
})

equity_df = equity_df.set_index(
    'Date'
)

st.line_chart(
    equity_df
)


# -------------------------------------------------
# EQUITY CURVE DATA
# -------------------------------------------------
st.subheader(
    '📋 Equity Curve Data'
)

st.dataframe(
    equity_df,
    width='stretch'
)


# -------------------------------------------------
# DAILY TOP RANKED STOCKS
# -------------------------------------------------
st.subheader(
    '🏆 Daily Top Ranked Stocks'
)

selected_history = results[
    'selected_history'
]

selected_df = pd.DataFrame(
    selected_history
)

st.dataframe(
    selected_df,
    width='stretch'
)


# -------------------------------------------------
# FEATURE SET
# -------------------------------------------------
st.subheader(
    '🧬 Active Feature Set'
)

feature_df = pd.DataFrame({

    'Features': FEATURES
})

st.dataframe(
    feature_df,
    width='stretch'
)

