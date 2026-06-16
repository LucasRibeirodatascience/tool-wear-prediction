from __future__ import annotations

from math import sqrt

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, RandomForestRegressor, StackingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.svm import SVR


def build_model_registry(random_state: int = 42, include_optional: bool = True) -> dict[str, object]:
    models: dict[str, object] = {
        "ExtraTrees": ExtraTreesRegressor(
            n_estimators=300,
            random_state=random_state,
            min_samples_leaf=2,
        ),
        "RandomForest": RandomForestRegressor(
            n_estimators=300,
            random_state=random_state,
            min_samples_leaf=2,
        ),
        "GradientBoosting": GradientBoostingRegressor(random_state=random_state),
        "SVR": SVR(C=10.0, epsilon=0.01, kernel="rbf"),
    }

    if include_optional:
        try:
            from xgboost import XGBRegressor

            models["XGBoost"] = XGBRegressor(
                n_estimators=300,
                learning_rate=0.05,
                max_depth=3,
                subsample=0.8,
                colsample_bytree=0.8,
                objective="reg:squarederror",
                random_state=random_state,
            )
        except ImportError:
            pass

        try:
            from lightgbm import LGBMRegressor

            models["LightGBM"] = LGBMRegressor(
                n_estimators=300,
                learning_rate=0.05,
                random_state=random_state,
                verbose=-1,
            )
        except ImportError:
            pass

    return models


def add_stacking_model(models: dict[str, object], random_state: int = 42) -> dict[str, object]:
    base_names = [name for name in ["ExtraTrees", "RandomForest", "GradientBoosting", "SVR"] if name in models]
    estimators = [(name.lower(), clone(models[name])) for name in base_names]

    if len(estimators) < 2:
        return models

    models = dict(models)
    models["Stacking"] = StackingRegressor(
        estimators=estimators,
        final_estimator=Ridge(alpha=1.0, random_state=random_state),
        cv=5,
        passthrough=False,
    )
    return models


def evaluate_predictions(y_true: pd.Series | np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(sqrt(mean_squared_error(y_true, y_pred))),
    }


def cross_validate_models(
    X: pd.DataFrame,
    y: pd.Series,
    preprocessor,
    models: dict[str, object],
    n_splits: int = 5,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if len(X) < n_splits:
        raise ValueError("O numero de amostras deve ser maior ou igual ao numero de folds.")

    cv = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    metrics_rows: list[dict[str, float | str]] = []
    predictions = pd.DataFrame({"real": y.reset_index(drop=True)})

    for model_name, model in models.items():
        estimator = Pipeline(
            steps=[
                ("preprocessor", clone(preprocessor)),
                ("model", clone(model)),
            ]
        )

        y_pred = cross_val_predict(estimator, X, y, cv=cv)
        metrics = evaluate_predictions(y, y_pred)

        metrics_rows.append({"model": model_name, **metrics})
        predictions[model_name] = y_pred

    metrics_df = pd.DataFrame(metrics_rows).sort_values("r2", ascending=False)
    return metrics_df, predictions


def fit_pipeline(X: pd.DataFrame, y: pd.Series, preprocessor, model) -> Pipeline:
    pipeline = Pipeline(
        steps=[
            ("preprocessor", clone(preprocessor)),
            ("model", clone(model)),
        ]
    )
    pipeline.fit(X, y)
    return pipeline
