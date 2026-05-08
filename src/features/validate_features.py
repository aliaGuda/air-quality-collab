from pathlib import Path
import sys
import yaml
import pandas as pd


def load_params(path: str = "configs/params.yaml") -> dict:
    with open(path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def validate_file_exists(path: str) -> None:
    if not Path(path).exists():
        raise FileNotFoundError(f"Required file not found: {path}")


def validate_min_rows(df: pd.DataFrame, min_rows: int) -> None:
    if len(df) < min_rows:
        raise ValueError(
            f"Dataset has too few rows. Found {len(df)}, required at least {min_rows}."
        )


def validate_target_column(df: pd.DataFrame, target_column: str) -> None:
    if target_column not in df.columns:
        raise ValueError(f"Target column missing: {target_column}")


def validate_missing_values(df: pd.DataFrame, max_missing_share: float) -> None:
    missing_share = df.isna().mean()

    invalid_columns = missing_share[missing_share > max_missing_share]

    if not invalid_columns.empty:
        raise ValueError(
            "Columns exceed maximum missing-value share: "
            f"{invalid_columns.to_dict()}"
        )


def validate_no_duplicate_columns(df: pd.DataFrame) -> None:
    duplicated = df.columns[df.columns.duplicated()].tolist()

    if duplicated:
        raise ValueError(f"Duplicate columns found: {duplicated}")


def validate_numeric_features(df: pd.DataFrame, target_column: str) -> None:
    feature_df = df.drop(columns=[target_column], errors="ignore")

    numeric_cols = feature_df.select_dtypes(include=["int64", "float64"]).columns

    if len(numeric_cols) == 0:
        raise ValueError("No numeric feature columns found.")


def validate_required_columns(df: pd.DataFrame, required_columns: list[str]) -> None:
    missing_columns = [col for col in required_columns if col not in df.columns]

    if missing_columns:
        raise ValueError(f"Required columns missing: {missing_columns}")


def run_validation(data_path: str, params: dict) -> None:
    validate_file_exists(data_path)

    df = pd.read_csv(data_path)

    target_column = params["data"]["target_column"]
    min_rows = params["feature_validation"]["min_rows"]
    max_missing_share = params["feature_validation"]["max_missing_share"]

    required_columns = params["feature_validation"].get("required_columns", [])

    validate_no_duplicate_columns(df)
    validate_min_rows(df, min_rows)
    validate_target_column(df, target_column)
    validate_missing_values(df, max_missing_share)
    validate_numeric_features(df, target_column)

    if required_columns:
        validate_required_columns(df, required_columns)

    print(f"Feature validation passed for: {data_path}")
    print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")


def main() -> None:
    params = load_params()

    if not params.get("feature_validation", {}).get("enabled", True):
        print("Feature validation is disabled.")
        return

    paths_to_validate = [
        params["data"]["train_path"],
        params["data"]["test_path"],
    ]

    for path in paths_to_validate:
        run_validation(path, params)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"Feature validation failed: {error}")
        sys.exit(1)