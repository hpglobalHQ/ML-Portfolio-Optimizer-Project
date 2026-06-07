# validation.py

import numpy as np
import pandas as pd

from scipy.stats import spearmanr

from xgboost import XGBRanker

from model import FEATURES


# -------------------------------------------------
# SETTINGS
# -------------------------------------------------
TOP_N = 5

INITIAL_CAPITAL = 100000

TRANSACTION_COST = 0.002

REBALANCE_FREQUENCY = 20

PREDICTION_STD_THRESHOLD = 0.00005


# -------------------------------------------------
# WALK-FORWARD VALIDATION
# -------------------------------------------------
def walk_forward_validation(
    df,
    market_df,
    trained_model,
    top_n=TOP_N,
    confidence_threshold=1.0,
    rebalance_frequency=REBALANCE_FREQUENCY,
    debug=True
):

    # -------------------------------------------------
    # DATE-BASED OUT-OF-SAMPLE SPLIT
    # -------------------------------------------------
    dates = sorted(
        df.index.unique()
    )

    split_date = dates[
        int(len(dates) * 0.70)
    ]

    test = (
        df[
            df.index >= split_date
        ]
        .copy()
    )

    # -------------------------------------------------
    # ROLLING WALK-FORWARD PREDICTIONS
    # -------------------------------------------------
    rolling_predictions = []

    rolling_dates = sorted(
        test.index.unique()
    )

    rolling_model = None

    # -------------------------------------------------
    # WALK-FORWARD LOOP
    # -------------------------------------------------
    for i, current_date in enumerate(rolling_dates):

        # -------------------------------------------------
        # RETRAIN EVERY 20 DAYS
        # -------------------------------------------------
        if i % rebalance_frequency == 0 or rolling_model is None:

            train_data = df[
                df.index < current_date
            ]

            if len(train_data) < 5000:
                continue

            X_train = train_data[
                FEATURES
            ]

            y_train = train_data[
                'Target'
            ]

            groups = (

                train_data
                .groupby(train_data.index)
                .size()
                .values
            )

            # -------------------------------------------------
            # FRESH MODEL
            # -------------------------------------------------
            rolling_model = XGBRanker(

                n_estimators=150,

                max_depth=4,

                learning_rate=0.02,

                subsample=0.8,

                colsample_bytree=0.8,

                reg_alpha=0.1,

                reg_lambda=1.0,

                min_child_weight=3,

                gamma=0.1,

                objective='rank:pairwise',

                tree_method='hist',

                random_state=42,

                n_jobs=-1
            )

            rolling_model.fit(

                X_train,

                y_train,

                group=groups
            )

        # -------------------------------------------------
        # PREDICT EVERY DAY
        # -------------------------------------------------
        predict_data = test[
            test.index == current_date
        ].copy()

        if len(predict_data) == 0:
            continue

        X_predict = predict_data[
            FEATURES
        ]

        preds = rolling_model.predict(
            X_predict
        )

        # -------------------------------------------------
        # RANK NORMALIZATION
        # -------------------------------------------------
        preds = (

            pd.Series(preds)

            .rank(pct=True)

            .values
        )

        predict_data[
            'prediction'
        ] = preds

        rolling_predictions.append(
            predict_data
        )

    # -------------------------------------------------
    # CONCAT PREDICTIONS
    # -------------------------------------------------
    test = pd.concat(
        rolling_predictions
    )

    # -------------------------------------------------
    # DAILY IC CALCULATION
    # -------------------------------------------------
    daily_ics = []

    for date in sorted(
        test.index.unique()
    ):

        day_df = test.loc[date]

        if isinstance(day_df, pd.Series):
            continue

        if len(day_df) < 5:
            continue

        ic = spearmanr(

            day_df['prediction'],

            day_df['Target']

        )[0]

        if not np.isnan(ic):

            daily_ics.append(ic)

    avg_daily_ic = (

        np.mean(daily_ics)

        if len(daily_ics) > 0

        else 0
    )

    # -------------------------------------------------
    # ICIR
    # -------------------------------------------------
    icir = (

        np.mean(daily_ics)

        /

        (
            np.std(daily_ics)
            + 1e-6
        )
    )

    # -------------------------------------------------
    # ROLLING IC
    # -------------------------------------------------
    rolling_ic = (

        pd.Series(daily_ics)
        .rolling(20)
        .mean()
    )

    # -------------------------------------------------
    # TRUE PANEL RETURNS
    # -------------------------------------------------
    test['daily_return'] = (

        test
        .groupby('Ticker')['Close']
        .pct_change()
    )

    # -------------------------------------------------
    # TRACKING
    # -------------------------------------------------
    equity = INITIAL_CAPITAL

    equity_curve = []

    equity_dates = []

    trade_returns = []

    portfolio_returns = []

    selected_history = []

    turnover_history = []

    previous_longs = set()

    previous_shorts = set()

    # -------------------------------------------------
    # UNIQUE DATES
    # -------------------------------------------------
    unique_dates = sorted(
        test.index.unique()
    )

    # -------------------------------------------------
    # MAIN LOOP
    # -------------------------------------------------
    for i, current_date in enumerate(
        unique_dates
    ):

        # -------------------------------------------------
        # REBALANCE FREQUENCY
        # -------------------------------------------------
        if i % rebalance_frequency != 0:
            continue

        # -------------------------------------------------
        # CURRENT DAY DATA
        # -------------------------------------------------
        day_df = test.loc[current_date]

        # -------------------------------------------------
        # SAFETY CHECKS
        # -------------------------------------------------
        if isinstance(day_df, pd.Series):
            continue

        if len(day_df) < top_n:
            continue

        day_df = day_df.replace(
            [np.inf, -np.inf],
            np.nan
        )

        day_df = day_df.dropna()

        if len(day_df) < top_n:
            continue

        # -------------------------------------------------
        # MARKET REGIME FILTER
        # -------------------------------------------------
        spy_trend = (

            market_df['Close']

            /

            market_df['Close']
            .rolling(200)
            .mean()

        ).loc[current_date]

        bull_market = spy_trend > 1
        # -------------------------------------------------
        # PREDICTION DISPERSION FILTER
        # -------------------------------------------------
        daily_prediction_std = (

            day_df[
                'prediction'
            ]
            .std()
        )

        print(
            'Daily Prediction Std:',
            round(
                float(
                    daily_prediction_std
                ),
                6
            )
        )

        if (
            daily_prediction_std
            <
            PREDICTION_STD_THRESHOLD
        ):

            print(
                'SKIPPED DAY -> LOW DISPERSION'
            )

            continue

        # -------------------------------------------------
        # SECTOR NEUTRALIZATION
        # -------------------------------------------------
        sector_mean = (

            day_df
            .groupby('Sector')[
                'prediction'
            ]
            .transform('mean')
        )

        day_df[
            'neutral_prediction'
        ] = (

            day_df['prediction']
            - sector_mean
        )

        # -------------------------------------------------
        # CONFIDENCE SCORE
        # -------------------------------------------------
        day_df['confidence_score'] = (

            np.abs(
                day_df[
                    'neutral_prediction'
                ]
            )

            /

            (
                daily_prediction_std
                + 1e-6
            )
        )

        # -------------------------------------------------
        # CONFIDENCE FILTER
        # -------------------------------------------------
        day_df = day_df[

            day_df[
                'confidence_score'
            ]
            >
            confidence_threshold
        ]

        if len(day_df) < top_n:
            continue

        # -------------------------------------------------
        # SORT PREDICTIONS
        # -------------------------------------------------
        day_df = day_df.sort_values(

            'neutral_prediction',

            ascending=False
        )

        # -------------------------------------------------
        # DEBUG TOP PREDICTIONS
        # -------------------------------------------------
        if debug and i % 20 == 0:

            print(
                '\nTOP PREDICTIONS'
            )

            print(

                day_df[[
                    'Ticker',
                    'prediction',
                    'neutral_prediction',
                    'confidence_score'
                ]]
                .head(10)
            )

        # -------------------------------------------------
        # LONG POSITIONS
        # -------------------------------------------------
        if bull_market:

            long_positions = (

                day_df
                .groupby('Sector')
                .head(1)
                .head(top_n)
                .copy()
            )

        else:

            long_positions = (

                day_df
                .groupby('Sector')
                .head(1)
                .head(1)
                .copy()
            )

        # -------------------------------------------------
        # SHORT POSITIONS
        # -------------------------------------------------
        short_positions = (

            day_df
            .sort_values(
                'neutral_prediction',
                ascending=True
            )

            .groupby('Sector')
            .head(1)
            .head(top_n)
            .copy()
        )

        if len(long_positions) == 0:
            continue

        if len(short_positions) == 0:
            continue

        # -------------------------------------------------
        # VOLATILITY TARGETING
        # -------------------------------------------------
        long_positions['risk_weight'] = (

            np.abs(
                long_positions[
                    'neutral_prediction'
                ]
            )

            /

            np.sqrt(

                long_positions[
                    'volatility_20'
                ]
                + 1e-6
            )
        )

        short_positions['risk_weight'] = (

            np.abs(
                short_positions[
                    'neutral_prediction'
                ]
            )

            /

            np.sqrt(

                short_positions[
                    'volatility_20'
                ]
                + 1e-6
            )
        )
        # -------------------------------------------------
        # NORMALIZE WEIGHTS
        # -------------------------------------------------
        long_positions['risk_weight'] /= (

            long_positions[
                'risk_weight'
            ].sum()
        )

        short_positions['risk_weight'] /= (

            short_positions[
                'risk_weight'
            ].sum()
        )

        # -------------------------------------------------
        # POSITION WEIGHT CAP
        # -------------------------------------------------
        long_positions['risk_weight'] = np.minimum(

            long_positions['risk_weight'],

            0.40
        )

        short_positions['risk_weight'] = np.minimum(

            short_positions['risk_weight'],

            0.40
        )

        # -------------------------------------------------
        # RENORMALIZE
        # -------------------------------------------------
        long_positions['risk_weight'] /= (

            long_positions[
                'risk_weight'
            ].sum()
        )

        short_positions['risk_weight'] /= (

            short_positions[
                'risk_weight'
            ].sum()
        )

        # -------------------------------------------------
        # TURNOVER ANALYSIS
        # -------------------------------------------------
        current_longs = set(
            long_positions['Ticker']
        )

        current_shorts = set(
            short_positions['Ticker']
        )

        long_turnover = len(

            current_longs
            -
            previous_longs
        )

        short_turnover = len(

            current_shorts
            -
            previous_shorts
        )

        turnover = (

            long_turnover
            +
            short_turnover

        ) / (2 * top_n)

        turnover_history.append(
            turnover
        )

        previous_longs = current_longs

        previous_shorts = current_shorts

        # -------------------------------------------------
        # STORE PORTFOLIO HISTORY
        # -------------------------------------------------
        selected_history.append({

            'date': current_date,

            'longs': list(
                long_positions['Ticker']
            ),

            'shorts': list(
                short_positions['Ticker']
            )
        })

        # -------------------------------------------------
        # DEBUG
        # -------------------------------------------------
        if debug and i % 20 == 0:

            print(
                '\nLONG POSITIONS'
            )

            print(

                long_positions[[
                    'Ticker',
                    'Sector',
                    'prediction',
                    'neutral_prediction',
                    'risk_weight'
                ]]
            )

            print(
                '\nSHORT POSITIONS'
            )

            print(

                short_positions[[
                    'Ticker',
                    'Sector',
                    'prediction',
                    'neutral_prediction',
                    'risk_weight'
                ]]
            )

        # -------------------------------------------------
        # LONG RETURN
        # -------------------------------------------------
        long_return = 0

        # -------------------------------------------------
        # LONG LOOP
        # -------------------------------------------------
        for _, row in (
            long_positions.iterrows()
        ):

            future_returns = test[
                test['Ticker']
                ==
                row['Ticker']
            ]

            future_returns = (

                future_returns[
                    future_returns.index
                    >
                    current_date
                ]
                .iloc[
                    :
                    rebalance_frequency
                ]
            )

            if len(future_returns) == 0:
                continue

            stock_return = (

                (
                    1
                    +
                    future_returns[
                        'daily_return'
                    ]
                )
                .prod()
                - 1
            )

            weighted_return = (

                stock_return

                *

                row[
                    'risk_weight'
                ]
            )

            long_return += (
                weighted_return
            )

            trade_returns.append(
                weighted_return
            )

        # -------------------------------------------------
        # SHORT RETURN
        # -------------------------------------------------
        short_return = 0

        # -------------------------------------------------
        # SHORT LOOP
        # -------------------------------------------------
        for _, row in (
            short_positions.iterrows()
        ):

            future_returns = test[
                test['Ticker']
                ==
                row['Ticker']
            ]

            future_returns = (

                future_returns[
                    future_returns.index
                    >
                    current_date
                ]
                .iloc[
                    :
                    rebalance_frequency
                ]
            )

            if len(future_returns) == 0:
                continue

            stock_return = (

                (
                    1
                    +
                    future_returns[
                        'daily_return'
                    ]
                )
                .prod()
                - 1
            )

            weighted_return = (

                -stock_return

                *

                row[
                    'risk_weight'
                ]
            )

            short_return += (
                weighted_return
            )

        # -------------------------------------------------
        # MARKET NEUTRAL RETURN
        # -------------------------------------------------
        portfolio_return = (

            long_return
            +
            short_return
        )

        # -------------------------------------------------
        # TRANSACTION COSTS
        # -------------------------------------------------
        portfolio_return -= (
            TRANSACTION_COST
        )

        # -------------------------------------------------
        # STORE PORTFOLIO RETURN
        # -------------------------------------------------
        portfolio_returns.append(
            portfolio_return
        )

        # -------------------------------------------------
        # UPDATE EQUITY
        # -------------------------------------------------
        equity *= (
            1 + portfolio_return
        )

        # -------------------------------------------------
        # STORE EQUITY CURVE
        # -------------------------------------------------
        equity_curve.append(
            equity
        )

        equity_dates.append(
            current_date
        )

    # -------------------------------------------------
    # EQUITY SERIES
    # -------------------------------------------------
    equity_series = pd.Series(

        equity_curve,

        index=equity_dates
    )

    portfolio_returns = np.array(
        portfolio_returns
    )

    trade_returns = np.array(
        trade_returns
    )

    # -------------------------------------------------
    # TOTAL RETURN
    # -------------------------------------------------
    total_return = (

        equity
        /
        INITIAL_CAPITAL
        - 1
    )

    # -------------------------------------------------
    # ELAPSED DAYS
    # -------------------------------------------------
    if len(equity_dates) > 1:

        elapsed_days = (

            equity_dates[-1]
            -
            equity_dates[0]

        ).days

    else:

        elapsed_days = 1

    # -------------------------------------------------
    # ANNUALIZED RETURN
    # -------------------------------------------------
    annualized_return = (

        (1 + total_return)

        **

        (
            252
            /
            max(elapsed_days, 1)
        )

        - 1
    )

    # -------------------------------------------------
    # SHARPE RATIO
    # -------------------------------------------------
    if np.std(portfolio_returns) > 0:

        sharpe_ratio = (

            np.mean(portfolio_returns)

            /

            np.std(portfolio_returns)

        ) * np.sqrt(12)

    else:

        sharpe_ratio = 0

    # -------------------------------------------------
    # SORTINO RATIO
    # -------------------------------------------------
    downside_returns = (

        portfolio_returns[
            portfolio_returns < 0
        ]
    )

    if len(downside_returns) > 0:

        downside_std = (
            np.std(downside_returns)
        )

        if downside_std > 0:

            sortino_ratio = (

                np.mean(portfolio_returns)

                /

                downside_std

            ) * np.sqrt(12)

        else:

            sortino_ratio = 0

    else:

        sortino_ratio = 0

    # -------------------------------------------------
    # WIN RATE
    # -------------------------------------------------
    if len(trade_returns) > 0:

        win_rate = np.mean(
            trade_returns > 0
        )

    else:

        win_rate = 0

    # -------------------------------------------------
    # PORTFOLIO WIN RATE
    # -------------------------------------------------
    if len(portfolio_returns) > 0:

        portfolio_win_rate = np.mean(
            np.array(portfolio_returns) > 0
        )

    else:

        portfolio_win_rate = 0

    # -------------------------------------------------
    # AVERAGE RETURN
    # -------------------------------------------------
    if len(trade_returns) > 0:

        avg_return = np.mean(
            trade_returns
        )

    else:

        avg_return = 0

    # -------------------------------------------------
    # MAX DRAWDOWN
    # -------------------------------------------------
    running_peak = (
        equity_series.cummax()
    )

    drawdown = (

        equity_series
        -
        running_peak

    ) / running_peak

    max_drawdown = (
        drawdown.min()
    )

    # -------------------------------------------------
    # SIGNAL QUALITY
    # -------------------------------------------------
    prediction_std = float(
        np.std(
            test['prediction']
        )
    )

    prediction_mean = float(
        np.mean(
            test['prediction']
        )
    )

    signal_rate = (

        len(portfolio_returns)

        /

        max(
            len(unique_dates),
            1
        )
    )

    # -------------------------------------------------
    # TURNOVER
    # -------------------------------------------------
    avg_turnover = (

        np.mean(turnover_history)

        if len(turnover_history) > 0

        else 0
    )

    # -------------------------------------------------
    # BENCHMARK
    # -------------------------------------------------
    benchmark_return = (

        market_df[
            'SPY_Close'
        ].iloc[-1]

        /

        market_df[
            'SPY_Close'
        ].iloc[0]

        - 1
    )

    # -------------------------------------------------
    # EXCESS RETURN
    # -------------------------------------------------
    excess_return = (
        total_return
        -
        benchmark_return
    )

    # -------------------------------------------------
    # DISTRIBUTION
    # -------------------------------------------------
    median_return = (

        np.median(trade_returns)

        if len(trade_returns) > 0

        else 0
    )

    top_10_percentile = (

        np.percentile(
            trade_returns,
            90
        )

        if len(trade_returns) > 0

        else 0
    )

    bottom_10_percentile = (

        np.percentile(
            trade_returns,
            10
        )

        if len(trade_returns) > 0

        else 0
    )

    # -------------------------------------------------
    # RESULTS
    # -------------------------------------------------
    results = {

        'total_return': float(total_return),

        'annualized_return': float(annualized_return),

        'sharpe_ratio': float(sharpe_ratio),

        'sortino_ratio': float(sortino_ratio),

        'win_rate': float(win_rate),

        'portfolio_win_rate': float(portfolio_win_rate),

        'avg_return': float(avg_return),

        'max_drawdown': float(max_drawdown),

        'prediction_std': float(prediction_std),

        'prediction_mean': float(prediction_mean),

        'signal_rate': float(signal_rate),

        'benchmark_return': float(benchmark_return),

        'excess_return': float(excess_return),

        'avg_daily_ic': float(avg_daily_ic),

        'icir': float(icir),

        'median_return': float(median_return),

        'top_10_percentile': float(top_10_percentile),

        'bottom_10_percentile': float(bottom_10_percentile),

        'equity_curve': equity_curve,

        'equity_dates': equity_dates,

        'selected_history': selected_history,

        'rolling_ic': rolling_ic,

        'avg_turnover': float(avg_turnover),

        'total_trades': int(
            len(trade_returns)
        )
    }

    return results