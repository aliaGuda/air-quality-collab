import os
from typing import Tuple

import joblib
import pandas as pd
import yaml
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures, StandardScaler


def load_params(path: str = "configs/params.yaml") -> dict:
    with open(path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def split_features_target(
    df: pd.DataFrame, target_column: str
) -> Tuple[pd.DataFrame, pd.Series]:
    if target_column not in df.columns:
        raise ValueError(f"Target column not found: {target_column}")

    X = df.drop(columns=[target_column])
    y = df[target_column]

    return X, y


def get_feature_types(X: pd.DataFrame) -> Tuple[list[str], list[str]]:
    numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_features = X.select_dtypes(
        include=["object", "category", "bool"]
    ).columns.tolist()

    return numeric_features, categorical_features


def build_numeric_pipeline(params: dict) -> Pipeline:
    steps = [
        (
            "imputer",
            SimpleImputer(strategy=params["preprocessing"]["imputation_strategy"]),
        ),
        ("scaler", StandardScaler()),
    ]

    poly_cfg = params["preprocessing"]["polynomial_features"]

    if poly_cfg["enabled"]:
        steps.append(
            (
                "polynomial_features",
                PolynomialFeatures(
                    degree=poly_cfg["degree"],
                    include_bias=poly_cfg["include_bias"],
                ),
            )
        )

    return Pipeline(steps=steps)


def build_categorical_pipeline(params: dict) -> Pipeline:
    return Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy=params["preprocessing"][
                        "categorical_imputation_strategy"
                    ]
                ),
            ),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )


def build_preprocessing_pipeline(X: pd.DataFrame, params: dict) -> Pipeline:
    numeric_features, categorical_features = get_feature_types(X)

    numeric_pipeline = build_numeric_pipeline(params)
    categorical_pipeline = build_categorical_pipeline(params)

    column_transformer = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_features),
            ("categorical", categorical_pipeline, categorical_features),
        ],
        remainder="drop",
    )

    steps = [("preprocessor", column_transformer)]

    feature_selection_cfg = params["preprocessing"]["feature_selection"]

    if feature_selection_cfg["enabled"]:
        steps.append(
            (
                "feature_selection",
                SelectKBest(
                    score_func=f_regression,
                    k=feature_selection_cfg["k_best"],
                ),
            )
        )

    return Pipeline(steps=steps)


def save_processed_data(
    X_train_processed,
    X_test_processed,
    y_train: pd.Series,
    y_test: pd.Series,
) -> None:
    os.makedirs("data/processed", exist_ok=True)

    pd.DataFrame(X_train_processed).to_csv("data/processed/X_train.csv", index=False)
    pd.DataFrame(X_test_processed).to_csv("data/processed/X_test.csv", index=False)
    pd.DataFrame(y_train).to_csv("data/processed/y_train.csv", index=False)
    pd.DataFrame(y_test).to_csv("data/processed/y_test.csv", index=False)


def main() -> None:
    params = load_params()

    train_path = params["data"]["train_path"]
    test_path = params["data"]["test_path"]
    target_column = params["data"]["target_column"]
    pipeline_path = params["serving"]["preprocessing_pipeline_path"]

    os.makedirs(os.path.dirname(pipeline_path), exist_ok=True)

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    X_train, y_train = split_features_target(train_df, target_column)
    X_test, y_test = split_features_target(test_df, target_column)

    preprocessing_pipeline = build_preprocessing_pipeline(X_train, params)

    X_train_processed = preprocessing_pipeline.fit_transform(X_train, y_train)
    X_test_processed = preprocessing_pipeline.transform(X_test)

    save_processed_data(X_train_processed, X_test_processed, y_train, y_test)

    joblib.dump(preprocessing_pipeline, pipeline_path)

    print(f"Saved fitted preprocessing pipeline to: {pipeline_path}")
    print(f"X_train processed shape: {X_train_processed.shape}")
    print(f"X_test processed shape: {X_test_processed.shape}")


if __name__ == "__main__":
    main()