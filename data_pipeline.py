# data_pipeline.py

import yfinance as yf
import pandas as pd
import numpy as np
import requests

from io import StringIO

from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import as_completed


# -------------------------------------------------
# GET S&P500 TICKERS
# -------------------------------------------------
def get_sp500_tickers():

    url = (
        'https://en.wikipedia.org/wiki/'
        'List_of_S%26P_500_companies'
    )

    headers = {

        'User-Agent':

        (
            'Mozilla/5.0 '
            '(Windows NT 10.0; Win64; x64) '
            'AppleWebKit/537.36 '
            '(KHTML, like Gecko) '
            'Chrome/120.0 Safari/537.36'
        )
    }

    response = requests.get(

        url,

        headers=headers
    )

    tables = pd.read_html(

        StringIO(response.text)
    )

    table = tables[0]

    tickers = table[
        'Symbol'
    ].tolist()

    tickers = [

        ticker.replace('.', '-')

        for ticker in tickers
    ]

    return tickers


# -------------------------------------------------
# STOCK UNIVERSE
# -------------------------------------------------
UNIVERSE = get_sp500_tickers()


# -------------------------------------------------
# SETTINGS
# -------------------------------------------------
LOOKAHEAD = 20

START_DATE = '2018-01-01'


# -------------------------------------------------
# FUNDAMENTAL DOWNLOAD
# -------------------------------------------------
def download_fundamentals(
    ticker
):

    try:

        stock_info = yf.Ticker(
            ticker
        ).info

        return {

            'Ticker': ticker,

            'Sector':

            stock_info.get(
                'sector',
                'Unknown'
            ),

            'roe':

            stock_info.get(
                'returnOnEquity',
                np.nan
            ),

            'profit_margin':

            stock_info.get(
                'profitMargins',
                np.nan
            ),

            'revenue_growth':

            stock_info.get(
                'revenueGrowth',
                np.nan
            )
        }

    except:

        return {

            'Ticker': ticker,

            'Sector': 'Unknown',

            'roe': np.nan,

            'profit_margin': np.nan,

            'revenue_growth': np.nan
        }


# -------------------------------------------------
# FEATURE ENGINEERING
# -------------------------------------------------
def create_features(
    df,
    market_df
):

    # -------------------------------------------------
    # BASIC RETURNS
    # -------------------------------------------------
    df['Return'] = (

        df['Close']
        .pct_change(fill_method=None)
    )

    # -------------------------------------------------
    # MOVING AVERAGES
    # -------------------------------------------------
    for window in [20, 50, 100, 200]:

        df[f'SMA_{window}'] = (

            df['Close']
            .rolling(window)
            .mean()
        )

    # -------------------------------------------------
    # DISTANCE FROM SMA
    # -------------------------------------------------
    df['distance_sma20'] = (

        (
            df['Close']
            -
            df['SMA_20']
        )

        /

        (
            df['SMA_20']
            + 1e-6
        )
    )

    # -------------------------------------------------
    # MOMENTUM
    # -------------------------------------------------
    for window in [20, 50, 100]:

        df[f'momentum_{window}'] = (

            df['Close']

            /

            df['Close']
            .shift(window)

            - 1
        )

    # -------------------------------------------------
    # MEAN REVERSION
    # -------------------------------------------------
    df['mean_reversion_5'] = (

        df['Close']

        /

        (
            df['Close']
            .rolling(5)
            .mean()

            + 1e-6
        )

        - 1
    )

    # -------------------------------------------------
    # BREAKOUT
    # -------------------------------------------------
    rolling_high_20 = (

        df['High']
        .rolling(20)
        .max()
    )

    df['breakout_20'] = (

        df['Close']

        /

        (
            rolling_high_20
            + 1e-6
        )
    )

    # -------------------------------------------------
    # VOLATILITY
    # -------------------------------------------------
    df['volatility_20'] = (

        df['Return']
        .rolling(20)
        .std()
    )

    # -------------------------------------------------
    # MOMENTUM QUALITY
    # -------------------------------------------------
    df['momentum_quality'] = (

        df['momentum_20']

        /

        (
            df['volatility_20']
            + 1e-6
        )
    )

    # -------------------------------------------------
    # VOLUME TREND
    # -------------------------------------------------
    volume_sma20 = (

        df['Volume']
        .rolling(20)
        .mean()
    )

    df['volume_trend'] = (

        df['Volume']

        /

        (
            volume_sma20
            + 1e-6
        )
    )

    # -------------------------------------------------
    # MARKET FEATURES
    # -------------------------------------------------
    market_df['market_return'] = (

        market_df['Close']
        .pct_change(fill_method=None)
    )

    market_df['market_volatility_20'] = (

        market_df['market_return']
        .rolling(20)
        .std()
    )

    market_df['market_trend_50'] = (

        market_df['Close']

        /

        market_df['Close']
        .rolling(50)
        .mean()

        - 1
    )

    # -------------------------------------------------
    # ALIGN MARKET DATA
    # -------------------------------------------------
    df['market_return'] = (
        market_df['market_return']
    )

    df['market_volatility_20'] = (
        market_df['market_volatility_20']
    )

    df['market_trend_50'] = (
        market_df['market_trend_50']
    )

    # -------------------------------------------------
    # RELATIVE STRENGTH
    # -------------------------------------------------
    df['relative_strength_20'] = (

        df['momentum_20']

        -

        (
            market_df['Close']
            .pct_change(20)
        )
    )

    df['relative_strength_50'] = (

        df['momentum_50']

        -

        (
            market_df['Close']
            .pct_change(50)
        )
    )

    # -------------------------------------------------
    # BETA PROXY
    # -------------------------------------------------
    df['beta_proxy'] = (

        df['volatility_20']

        *

        df['relative_strength_20']
    )

    # -------------------------------------------------
    # INTERACTIONS
    # -------------------------------------------------
    df['momentum_volume_interaction'] = (

        df['momentum_20']

        *

        df['volume_trend']
    )

    df['relative_strength_trend_interaction'] = (

        df['relative_strength_20']

        *

        df['distance_sma20']
    )

    # -------------------------------------------------
    # RESIDUAL MOMENTUM
    # -------------------------------------------------
    df['residual_momentum_20'] = (

        df['momentum_20']

        -

        df['market_return']
        .rolling(20)
        .sum()
    )

    # -------------------------------------------------
    # CROSS-SECTIONAL FEATURES
    # -------------------------------------------------
    df['momentum_20_zscore'] = (

        (
            df['momentum_20']

            -

            df['momentum_20']
            .rolling(60)
            .mean()
        )

        /

        (
            df['momentum_20']
            .rolling(60)
            .std()

            + 1e-6
        )
    )

    # -------------------------------------------------
    # FEATURE COLUMNS
    # -------------------------------------------------
    feature_columns = [

        'distance_sma20',

        'momentum_20',
        'momentum_50',
        'momentum_100',

        'breakout_20',

        'mean_reversion_5',

        'volatility_20',

        'momentum_quality',

        'volume_trend',

        'market_return',
        'market_volatility_20',
        'market_trend_50',

        'relative_strength_20',
        'relative_strength_50',

        'beta_proxy',

        'momentum_volume_interaction',

        'relative_strength_trend_interaction',

        'momentum_20_zscore',

        'roe',

        'profit_margin',

        'revenue_growth'
    ]

    # -------------------------------------------------
    # LAG FEATURES TO PREVENT LEAKAGE
    # -------------------------------------------------
    for col in feature_columns:

        df[col] = df[col].shift(1)

    return df


# -------------------------------------------------
# LOAD DATA
# -------------------------------------------------
def load_cross_sectional_data():

    # -------------------------------------------------
    # DOWNLOAD MARKET DATA
    # -------------------------------------------------
    market_df = yf.download(

        'SPY',

        start=START_DATE,

        progress=False,

        auto_adjust=True
    )

    # -------------------------------------------------
    # FLAT COLUMNS
    # -------------------------------------------------
    market_df.columns = [

        col[0]

        if isinstance(col, tuple)

        else col

        for col in market_df.columns
    ]

    # -------------------------------------------------
    # REMOVE DUPLICATES
    # -------------------------------------------------
    market_df = market_df.loc[
        :,
        ~market_df.columns.duplicated()
    ]

    # -------------------------------------------------
    # KEEP CLOSE
    # -------------------------------------------------
    market_df = (
        market_df[['Close']]
        .copy()
    )

    market_df['SPY_Close'] = (
        market_df['Close']
    )

    # -------------------------------------------------
    # BATCH DOWNLOAD S&P500
    # -------------------------------------------------
    print(
        'DOWNLOADING S&P500 DATA...'
    )

    batch_data = yf.download(

        UNIVERSE,

        start=START_DATE,

        progress=True,

        auto_adjust=True,

        group_by='ticker',

        threads=True
    )

    # -------------------------------------------------
    # PARALLEL FUNDAMENTAL DOWNLOAD
    # -------------------------------------------------
    print(
        'DOWNLOADING FUNDAMENTALS...'
    )

    fundamental_cache = {}

    with ThreadPoolExecutor(
        max_workers=16
    ) as executor:

        futures = {

            executor.submit(

                download_fundamentals,

                ticker

            ): ticker

            for ticker in UNIVERSE
        }

        for future in as_completed(
            futures
        ):

            result = future.result()

            ticker = result[
                'Ticker'
            ]

            fundamental_cache[
                ticker
            ] = result

    # -------------------------------------------------
    # STORAGE
    # -------------------------------------------------
    all_data = []

    # -------------------------------------------------
    # LOOP THROUGH TICKERS
    # -------------------------------------------------
    for ticker in UNIVERSE:

        try:

            print(
                f'PROCESSING: {ticker}'
            )

            # -------------------------------------------------
            # EXTRACT TICKER DATA
            # -------------------------------------------------
            df = batch_data[
                ticker
            ].copy()

            # -------------------------------------------------
            # EMPTY CHECK
            # -------------------------------------------------
            if len(df) < 300:
                continue

            # -------------------------------------------------
            # FLAT COLUMNS
            # -------------------------------------------------
            df.columns = [

                str(col)

                for col in df.columns
            ]

            # -------------------------------------------------
            # REQUIRED COLUMNS
            # -------------------------------------------------
            required_columns = [

                'Open',
                'High',
                'Low',
                'Close',
                'Volume'
            ]

            # -------------------------------------------------
            # MISSING CHECK
            # -------------------------------------------------
            missing_columns = [

                col

                for col in required_columns

                if col not in df.columns
            ]

            if len(missing_columns) > 0:

                print(
                    f'MISSING: {ticker}'
                )

                continue

            # -------------------------------------------------
            # KEEP PRICE COLUMNS
            # -------------------------------------------------
            df = df[
                required_columns
            ].copy()

            # -------------------------------------------------
            # FORCE NUMERIC
            # -------------------------------------------------
            for col in required_columns:

                df[col] = pd.to_numeric(

                    df[col],

                    errors='coerce'
                )

            # -------------------------------------------------
            # METADATA
            # -------------------------------------------------
            df['Ticker'] = ticker

            df['Sector'] = (

                fundamental_cache[ticker]['Sector']
            )

            df['roe'] = (

                fundamental_cache[ticker]['roe']
            )

            df['profit_margin'] = (

                fundamental_cache[ticker]['profit_margin']
            )

            df['revenue_growth'] = (

                fundamental_cache[ticker]['revenue_growth']
            )

            # -------------------------------------------------
            # FEATURES
            # -------------------------------------------------
            df = create_features(

                df,

                market_df
            )

            # -------------------------------------------------
            # FUTURE RETURNS
            # -------------------------------------------------
            future_return = (

                df['Close']
                .shift(-LOOKAHEAD)

                /

                df['Close']

                - 1
            )

            # -------------------------------------------------
            # MARKET FUTURE RETURNS
            # -------------------------------------------------
            market_future_return = (

                market_df['Close']
                .shift(-LOOKAHEAD)

                /

                market_df['Close']

                - 1
            )

            # -------------------------------------------------
            # TARGET
            # -------------------------------------------------
            df['Target'] = (

                future_return

                -

                market_future_return
            )

            # -------------------------------------------------
            # CLEAN
            # -------------------------------------------------
            df = df.replace(
                [np.inf, -np.inf],
                np.nan
            )

            df = df.dropna()

            # -------------------------------------------------
            # FINAL LENGTH CHECK
            # -------------------------------------------------
            if len(df) < 200:
                continue

            # -------------------------------------------------
            # APPEND
            # -------------------------------------------------
            all_data.append(df)

            print(
                f'LOADED: {ticker}'
            )

        except Exception as e:

            print(
                f'FAILED: {ticker}'
            )

            print(e)

    # -------------------------------------------------
    # CONCAT DATA
    # -------------------------------------------------
    full_df = pd.concat(
        all_data
    )

    # -------------------------------------------------
    # DATE INDEX
    # -------------------------------------------------
    full_df.index = pd.to_datetime(
        full_df.index
    )

    # -------------------------------------------------
    # SORT
    # -------------------------------------------------
    full_df = full_df.sort_index()

    # -------------------------------------------------
    # FINAL CLEANING
    # -------------------------------------------------
    full_df = full_df.replace(
        [np.inf, -np.inf],
        np.nan
    )

    full_df = full_df.dropna()

    # -------------------------------------------------
    # DEBUG
    # -------------------------------------------------
    print(
        '\nFINAL DATASET SHAPE:',
        full_df.shape
    )

    print(
        '\nTOTAL STOCKS:',
        full_df['Ticker'].nunique()
    )

    print(
        '\nFEATURE NULL COUNTS:\n'
    )

    print(
        full_df.isnull().sum()
    )

    return full_df, market_df