# model.py

import numpy as np
import pandas as pd

from scipy.stats import spearmanr

from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error
)

from xgboost import XGBRanker

from catboost import CatBoostRanker


# -------------------------------------------------
# FEATURE SET
# -------------------------------------------------
FEATURES = [

    # TREND
    'distance_sma20',

    # MOMENTUM
    'momentum_20',
    'momentum_50',
    'momentum_100',

    # BREAKOUT
    'breakout_20',

    # MEAN REVERSION
    'mean_reversion_5',

    # VOLATILITY
    'volatility_20',

    # QUALITY
    'momentum_quality',

    # VOLUME
    'volume_trend',

    # MARKET FEATURES
    'market_return',
    'market_volatility_20',
    'market_trend_50',

    # RELATIVE STRENGTH
    'relative_strength_20',
    'relative_strength_50',

    # BETA
    'beta_proxy',

    # INTERACTIONS
    'momentum_volume_interaction',

    'relative_strength_trend_interaction',

    # CROSS-SECTIONAL
    'momentum_20_zscore',

    # FUNDAMENTALS

    'roe',

    'profit_margin',

    'revenue_growth'
]


# -------------------------------------------------
# TRAIN MODEL
# -------------------------------------------------
def train_model(
    X,
    y,
    debug=True
):

    # -------------------------------------------------
    # VALIDATION SPLIT
    # -------------------------------------------------
    split_index = int(
        len(X) * 0.80
    )

    X_train = (
        X.iloc[:split_index]
        .copy()
    )

    y_train = (
        y.iloc[:split_index]
        .copy()
    )

    X_valid = (
        X.iloc[split_index:]
        .copy()
    )

    y_valid = (
        y.iloc[split_index:]
        .copy()
    )

    # -------------------------------------------------
    # CLEAN DATA
    # -------------------------------------------------
    X_train = X_train.replace(
        [np.inf, -np.inf],
        np.nan
    )

    X_valid = X_valid.replace(
        [np.inf, -np.inf],
        np.nan
    )

    X_train = X_train.fillna(0)

    X_valid = X_valid.fillna(0)

    y_train = y_train.fillna(0)

    y_valid = y_valid.fillna(0)

    # -------------------------------------------------
    # CROSS-SECTIONAL GROUPS
    # -------------------------------------------------
    train_groups = []

    current_date = None

    count = 0

    for idx in X_train.index:

        if current_date is None:

            current_date = idx

            count = 1

        elif idx == current_date:

            count += 1

        else:

            train_groups.append(
                count
            )

            current_date = idx

            count = 1

    train_groups.append(
        count
    )

    # -------------------------------------------------
    # VALID GROUPS
    # -------------------------------------------------
    valid_groups = []

    current_date = None

    count = 0

    for idx in X_valid.index:

        if current_date is None:

            current_date = idx

            count = 1

        elif idx == current_date:

            count += 1

        else:

            valid_groups.append(
                count
            )

            current_date = idx

            count = 1

    valid_groups.append(
        count
    )

    # -------------------------------------------------
    # XGBOOST RANKER
    # -------------------------------------------------
    xgb_model = XGBRanker(

        n_estimators=200,

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

    # -------------------------------------------------
    # CATBOOST RANKER
    # -------------------------------------------------
    cat_model = CatBoostRanker(

        iterations=200,

        learning_rate=0.02,

        depth=4,

        loss_function='YetiRank',

        random_seed=42,

        verbose=False
    )

    # -------------------------------------------------
    # TRAIN XGBOOST
    # -------------------------------------------------
    xgb_model.fit(

        X_train,

        y_train,

        group=train_groups
    )

    # -------------------------------------------------
    # TRAIN CATBOOST
    # -------------------------------------------------
    cat_model.fit(

        X_train,

        y_train,

        group_id=X_train.index.astype(str).to_numpy()
    )

    # -------------------------------------------------
    # VALIDATION PREDICTIONS
    # -------------------------------------------------
    xgb_preds = xgb_model.predict(
        X_valid
    )

    cat_preds = cat_model.predict(
        X_valid
    )

    # -------------------------------------------------
    # ENSEMBLE PREDICTIONS
    # -------------------------------------------------
    ensemble_preds = (

        xgb_preds

        +

        cat_preds

    ) / 2

    # -------------------------------------------------
    # VALIDATION METRICS
    # -------------------------------------------------
    mse = mean_squared_error(

        y_valid,

        ensemble_preds
    )

    mae = mean_absolute_error(

        y_valid,

        ensemble_preds
    )

    # -------------------------------------------------
    # INFORMATION COEFFICIENT
    # -------------------------------------------------
    ic = spearmanr(

        y_valid,

        ensemble_preds

    )[0]

    # -------------------------------------------------
    # DAILY IC
    # -------------------------------------------------
    valid_df = pd.DataFrame({

        'prediction': ensemble_preds,

        'target': y_valid.values

    }, index=X_valid.index)

    daily_ics = []

    for date in sorted(
        valid_df.index.unique()
    ):

        day_df = valid_df.loc[date]

        if isinstance(day_df, pd.Series):
            continue

        if len(day_df) < 5:
            continue

        day_ic = spearmanr(

            day_df['prediction'],

            day_df['target']

        )[0]

        if not np.isnan(day_ic):

            daily_ics.append(day_ic)

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
    # FEATURE IMPORTANCE
    # -------------------------------------------------
    feature_importance = (

        xgb_model
        .feature_importances_
    )

    # -------------------------------------------------
    # DEBUG OUTPUT
    # -------------------------------------------------
    if debug:

        print(
            '\nMODEL TRAINING COMPLETE'
        )

        print(
            '\nVALIDATION METRICS'
        )

        print(
            f'MSE: {mse:.6f}'
        )

        print(
            f'MAE: {mae:.6f}'
        )

        print(
            f'IC: {ic:.6f}'
        )

        print(
            f'AVG DAILY IC: {avg_daily_ic:.6f}'
        )

        print(
            f'ICIR: {icir:.6f}'
        )

        # -------------------------------------------------
        # FEATURE IMPORTANCE TABLE
        # -------------------------------------------------
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

        print(
            '\nFEATURE IMPORTANCE'
        )

        print(
            importance_df
        )

    # -------------------------------------------------
    # MODEL PACKAGE
    # -------------------------------------------------
    model_package = {

        'xgb_model': xgb_model,

        'cat_model': cat_model,

        'ensemble': True,

        'mse': float(mse),

        'mae': float(mae),

        'ic': float(ic),

        'avg_daily_ic': float(avg_daily_ic),

        'icir': float(icir),

        'feature_importance': feature_importance
    }

    return model_package