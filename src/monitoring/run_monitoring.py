import json
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml
from evidently import ColumnMapping
from evidently.metric_preset import DataDriftPreset, DataQualityPreset, RegressionPreset
from evidently.report import Report


REPORT_DIR = Path("src/monitoring/evidently_reports")
LOG_PATH = Path("src/monitoring/drift_warnings.log")

REFERENCE_PATH = Path("data/splits/reference.csv")
PRODUCTION_PATH = Path("data/splits/production.csv")
TEST_PATH = Path("data/splits/test.csv")

MODEL_PATH = Path("models/model.joblib")
PREPROCESSOR_PATH = Path("models/preprocessing_pipeline.joblib")
PARAMS_PATH = Path("configs/params.yaml")

DRIFT_THRESHOLD = 0.20

REPORT_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    filename=LOG_PATH,
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)


def load_params() -> dict:
    with open(PARAMS_PATH, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def load_dataset(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing dataset: {path}")

    df = pd.read_csv(path)

    if df.empty:
        raise ValueError(f"Dataset is empty: {path}")

    return df


def load_model_assets():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Missing model artifact: {MODEL_PATH}")

    if not PREPROCESSOR_PATH.exists():
        raise FileNotFoundError(f"Missing preprocessing artifact: {PREPROCESSOR_PATH}")

    model = joblib.load(MODEL_PATH)
    preprocessor = joblib.load(PREPROCESSOR_PATH)

    return model, preprocessor


def add_predictions(
    df: pd.DataFrame,
    model,
    preprocessor,
    target_column: str,
) -> pd.DataFrame:
    result_df = df.copy()

    if target_column not in result_df.columns:
        raise ValueError(f"Target column missing: {target_column}")

    X = result_df.drop(columns=[target_column])
    X_transformed = preprocessor.transform(X)

    result_df["prediction"] = model.predict(X_transformed)

    return result_df


def inject_drift(df: pd.DataFrame) -> pd.DataFrame:
    drifted_df = df.copy()

    drift_features = [
        "PT08.S1(CO)",
        "C6H6(GT)",
        "T",
        "RH",
    ]

    rng = np.random.default_rng(seed=42)

    if "PT08.S1(CO)" in drifted_df.columns:
        drifted_df["PT08.S1(CO)"] = drifted_df["PT08.S1(CO)"] * 1.65

    if "C6H6(GT)" in drifted_df.columns:
        drifted_df["C6H6(GT)"] = drifted_df["C6H6(GT)"] * 1.50

    if "T" in drifted_df.columns:
        drifted_df["T"] = drifted_df["T"] + 8

    if "RH" in drifted_df.columns:
        drifted_df["RH"] = drifted_df["RH"] + rng.normal(
            loc=15,
            scale=4,
            size=len(drifted_df),
        )

    actually_drifted = [col for col in drift_features if col in drifted_df.columns]
    print(f"Injected drift into features: {actually_drifted}")

    return drifted_df


def build_column_mapping(
    df: pd.DataFrame,
    target_column: str,
) -> ColumnMapping:
    numerical_features = [
        column
        for column in df.columns
        if column not in [target_column, "prediction"]
        and pd.api.types.is_numeric_dtype(df[column])
    ]

    column_mapping = ColumnMapping()
    column_mapping.target = target_column
    column_mapping.prediction = "prediction"
    column_mapping.numerical_features = numerical_features

    return column_mapping


def create_evidently_report(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    column_mapping: ColumnMapping,
    output_path: Path,
) -> dict:
    report = Report(
        metrics=[
            DataDriftPreset(),
            DataQualityPreset(),
            RegressionPreset(),
        ]
    )

    report.run(
        reference_data=reference_df,
        current_data=current_df,
        column_mapping=column_mapping,
    )

    report.save_html(str(output_path))

    return report.as_dict()


def extract_drift_summary(report_dict: dict) -> tuple[float, list[dict]]:
    drift_share = 0.0
    drifted_features = []

    for metric in report_dict.get("metrics", []):
        result = metric.get("result", {})

        if "share_of_drifted_columns" in result:
            drift_share = result["share_of_drifted_columns"]

        drift_by_columns = result.get("drift_by_columns", {})

        for feature_name, feature_result in drift_by_columns.items():
            if feature_result.get("drift_detected") is True:
                drifted_features.append(
                    {
                        "feature": feature_name,
                        "score": feature_result.get("drift_score"),
                        "method": feature_result.get("stattest_name"),
                    }
                )

    return drift_share, drifted_features


def apply_threshold_logic(
    report_name: str,
    report_dict: dict,
) -> None:
    drift_share, drifted_features = extract_drift_summary(report_dict)

    result = {
        "event": "DRIFT_CHECK_RESULT",
        "report": report_name,
        "drift_share": drift_share,
        "threshold": DRIFT_THRESHOLD,
        "drift_detected": drift_share > DRIFT_THRESHOLD,
        "drifted_features": drifted_features,
    }

    print(json.dumps(result, indent=2))

    if drift_share > DRIFT_THRESHOLD:
        logging.warning(json.dumps(result))
    else:
        logging.info(json.dumps(result))


def main() -> None:
    params = load_params()
    target_column = params["data"]["target_column"]

    model, preprocessor = load_model_assets()

    reference_df = load_dataset(REFERENCE_PATH)
    production_df = load_dataset(PRODUCTION_PATH)
    test_df = load_dataset(TEST_PATH)

    reference_df = add_predictions(
        reference_df,
        model,
        preprocessor,
        target_column,
    )

    clean_test_df = add_predictions(
        test_df,
        model,
        preprocessor,
        target_column,
    )

    production_df = add_predictions(
        production_df,
        model,
        preprocessor,
        target_column,
    )

    drifted_production_df = inject_drift(production_df)

    drifted_production_df = add_predictions(
        drifted_production_df.drop(columns=["prediction"]),
        model,
        preprocessor,
        target_column,
    )

    column_mapping = build_column_mapping(reference_df, target_column)

    baseline_report_dict = create_evidently_report(
        reference_df=reference_df,
        current_df=clean_test_df,
        column_mapping=column_mapping,
        output_path=REPORT_DIR / "baseline_report.html",
    )

    drift_report_dict = create_evidently_report(
        reference_df=reference_df,
        current_df=drifted_production_df,
        column_mapping=column_mapping,
        output_path=REPORT_DIR / "drift_report.html",
    )

    apply_threshold_logic("baseline_report", baseline_report_dict)
    apply_threshold_logic("drift_report", drift_report_dict)

    print("Monitoring completed successfully.")
    print(f"Baseline report saved to: {REPORT_DIR / 'baseline_report.html'}")
    print(f"Drift report saved to: {REPORT_DIR / 'drift_report.html'}")
    print(f"Drift warning log saved to: {LOG_PATH}")


if __name__ == "__main__":
    main()