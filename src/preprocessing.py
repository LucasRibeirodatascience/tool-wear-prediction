from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def load_dataset(data_path: str | Path) -> pd.DataFrame:
    path = Path(data_path)
    if not path.exists():
        raise FileNotFoundError(f"Arquivo de dados nao encontrado: {path}")

    if path.suffix.lower() != ".csv":
        raise ValueError("O pipeline espera um arquivo CSV.")

    return pd.read_csv(path)


def parse_features(features: str | Iterable[str] | None) -> list[str] | None:
    if features is None:
        return None

    if isinstance(features, str):
        parsed = [feature.strip() for feature in features.split(",")]
    else:
        parsed = [str(feature).strip() for feature in features]

    parsed = [feature for feature in parsed if feature]
    return parsed or None


def infer_numeric_features(data: pd.DataFrame, target: str) -> list[str]:
    numeric_columns = data.select_dtypes(include="number").columns.tolist()
    features = [column for column in numeric_columns if column != target]

    if not features:
        raise ValueError(
            "Nao foi possivel inferir variaveis numericas de entrada. "
            "Informe as colunas com --features."
        )

    return features


def split_features_target(
    data: pd.DataFrame,
    target: str = "Desgaste",
    features: str | Iterable[str] | None = None,
) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    if target not in data.columns:
        raise ValueError(f"Coluna alvo '{target}' nao encontrada no CSV.")

    selected_features = parse_features(features) or infer_numeric_features(data, target)
    missing_features = [feature for feature in selected_features if feature not in data.columns]

    if missing_features:
        missing = ", ".join(missing_features)
        raise ValueError(f"Variaveis ausentes no CSV: {missing}")

    X = data[selected_features].copy()
    y = data[target].copy()

    non_numeric = X.select_dtypes(exclude="number").columns.tolist()
    if non_numeric:
        columns = ", ".join(non_numeric)
        raise ValueError(
            "O pipeline aceita apenas variaveis numericas. "
            f"Colunas nao numericas encontradas: {columns}"
        )

    return X, y, selected_features


def build_preprocessor(feature_names: list[str]) -> ColumnTransformer:
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    return ColumnTransformer(
        transformers=[("numeric", numeric_pipeline, feature_names)],
        remainder="drop",
    )
