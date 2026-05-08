import os
import yaml
import pandas as pd


def load_params(path="configs/params.yaml"):
    with open(path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def read_air_quality_csv(raw_path: str) -> pd.DataFrame:
    # Try common separators because AirQualityUCI versions differ.
    for sep in [";", ","]:
        df = pd.read_csv(raw_path, sep=sep, engine="python")
        df.columns = df.columns.str.strip()

        if df.shape[1] > 1:
            return df

    raise ValueError(
        "Could not parse CSV correctly. File was read as one column. "
        "Check the dataset separator."
    )


def main():
    params = load_params()

    raw_path = params["data"]["raw_path"]
    cleaned_path = params["data"]["cleaned_path"]
    missing_marker = params["data"]["missing_value_marker"]

    os.makedirs(os.path.dirname(cleaned_path), exist_ok=True)

    df = read_air_quality_csv(raw_path)

    df = df.loc[:, ~df.columns.astype(str).str.contains("^Unnamed")]
    df.replace(missing_marker, pd.NA, inplace=True)
    df.dropna(how="all", inplace=True)
    df.reset_index(drop=True, inplace=True)

    df.to_csv(cleaned_path, index=False)

    print(f"Cleaned data saved to {cleaned_path}")
    print(f"Columns: {df.columns.tolist()}")
    print(f"Shape: {df.shape}")


if __name__ == "__main__":
    main()