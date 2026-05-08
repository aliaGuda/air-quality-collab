import os
import yaml
import pandas as pd
from sklearn.model_selection import train_test_split


def load_params(path="configs/params.yaml"):
    with open(path, "r") as file:
        return yaml.safe_load(file)


def add_datetime_features(df, date_col, time_col):
    if date_col in df.columns and time_col in df.columns:
        dt = pd.to_datetime(
            df[date_col].astype(str) + " " + df[time_col].astype(str),
            errors="coerce",
            dayfirst=True,
        )

        df["hour"] = dt.dt.hour
        df["day"] = dt.dt.day
        df["month"] = dt.dt.month
        df["day_of_week"] = dt.dt.dayofweek

        df = df.drop(columns=[date_col, time_col])

    return df


def main():
    params = load_params()

    cleaned_path = params["data"]["cleaned_path"]
    train_path = params["data"]["train_path"]
    test_path = params["data"]["test_path"]
    reference_path = params["data"]["reference_path"]
    production_path = params["data"]["production_path"]
    target_column = params["data"]["target_column"]
    test_size = params["data"]["test_size"]
    reference_size = params["data"]["reference_size"]
    random_state = params["project"]["random_state"]

    date_col = params["features"]["datetime_columns"]["date"]
    time_col = params["features"]["datetime_columns"]["time"]

    os.makedirs(os.path.dirname(train_path), exist_ok=True)

    df = pd.read_csv(cleaned_path)
    df = add_datetime_features(df, date_col, time_col)

    if target_column not in df.columns:
        raise ValueError(f"Target column not found: {target_column}")

    df = df.dropna(subset=[target_column])
    df = df.reset_index(drop=True)

    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        shuffle=True,
    )

    reference_df, production_df = train_test_split(
        test_df,
        test_size=1 - reference_size,
        random_state=random_state,
        shuffle=True,
    )

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    reference_df.to_csv(reference_path, index=False)
    production_df.to_csv(production_path, index=False)

    print(f"Train saved to {train_path}: {train_df.shape}")
    print(f"Test saved to {test_path}: {test_df.shape}")
    print(f"Reference saved to {reference_path}: {reference_df.shape}")
    print(f"Production saved to {production_path}: {production_df.shape}")


if __name__ == "__main__":
    main()