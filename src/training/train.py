import json
import os
import time
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import optuna
import pandas as pd
import yaml
from matplotlib import pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score


def load_params(path: str = "configs/params.yaml") -> dict:
    with open(path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def load_data() -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    X_train = pd.read_csv("data/processed/X_train.csv")
    X_test = pd.read_csv("data/processed/X_test.csv")
    y_train = pd.read_csv("data/processed/y_train.csv").squeeze("columns")
    y_test = pd.read_csv("data/processed/y_test.csv").squeeze("columns")
    return X_train, y_train, X_test, y_test


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def evaluate_model(model, X_test, y_test) -> dict:
    predictions = model.predict(X_test)

    return {
        "mae": float(mean_absolute_error(y_test, predictions)),
        "mse": float(mean_squared_error(y_test, predictions)),
        "rmse": rmse(y_test, predictions),
        "r2": float(r2_score(y_test, predictions)),
    }


def log_dataset_metadata(X_train, X_test) -> None:
    mlflow.log_param("dataset_name", "Air Quality UCI")
    mlflow.log_param("dataset_version", "v1")
    mlflow.log_param("train_rows", len(X_train))
    mlflow.log_param("test_rows", len(X_test))
    mlflow.log_param("num_features", X_train.shape[1])


def save_prediction_plot(y_true, y_pred, path: str) -> None:
    plt.figure()
    plt.scatter(y_true, y_pred, alpha=0.5)
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title("Actual vs Predicted")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def save_loss_curve(model_name: str, loss_values: list[float], path: str) -> None:
    plt.figure()
    plt.plot(range(1, len(loss_values) + 1), loss_values, marker="o")
    plt.xlabel("Trial / Step")
    plt.ylabel("Validation RMSE")
    plt.title(f"Loss Curve - {model_name}")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def get_baseline_models(random_state: int) -> dict:
    return {
        "ridge": Ridge(alpha=1.0),
        "random_forest": RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=random_state,
            n_jobs=-1,
        ),
        "gradient_boosting": GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=3,
            random_state=random_state,
        ),
    }


def create_model_from_trial(trial, algorithm: str, random_state: int):
    if algorithm == "ridge":
        return Ridge(
            alpha=trial.suggest_float("alpha", 0.001, 100.0, log=True)
        )

    if algorithm == "random_forest":
        return RandomForestRegressor(
            n_estimators=trial.suggest_int("n_estimators", 100, 300),
            max_depth=trial.suggest_int("max_depth", 4, 20),
            min_samples_split=trial.suggest_int("min_samples_split", 2, 10),
            min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 5),
            random_state=random_state,
            n_jobs=-1,
        )

    if algorithm == "gradient_boosting":
        return GradientBoostingRegressor(
            n_estimators=trial.suggest_int("n_estimators", 100, 300),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            max_depth=trial.suggest_int("max_depth", 2, 6),
            subsample=trial.suggest_float("subsample", 0.6, 1.0),
            random_state=random_state,
        )

    raise ValueError(f"Unsupported algorithm: {algorithm}")


def objective(trial, algorithm: str, X_train, y_train, random_state: int) -> float:
    model = create_model_from_trial(trial, algorithm, random_state)

    scores = cross_val_score(
        model,
        X_train,
        y_train,
        scoring="neg_root_mean_squared_error",
        cv=3,
        n_jobs=-1,
    )

    return float(-scores.mean())


def train_and_log_baseline_model(
    model_name: str,
    model,
    X_train,
    y_train,
    X_test,
    y_test,
) -> dict:
    artifact_dir = Path("artifacts") / model_name
    artifact_dir.mkdir(parents=True, exist_ok=True)

    start_time = time.time()

    with mlflow.start_run(run_name=model_name) as run:
        model.fit(X_train, y_train)

        training_time = time.time() - start_time
        predictions = model.predict(X_test)
        metrics = evaluate_model(model, X_test, y_test)

        mlflow.log_param("algorithm", model_name)
        mlflow.log_param("run_type", "baseline")
        mlflow.log_params(model.get_params())
        log_dataset_metadata(X_train, X_test)

        mlflow.log_metric("training_time_seconds", training_time)

        for metric_name, value in metrics.items():
            mlflow.log_metric(f"test_{metric_name}", value)

        loss_values = []
        for fraction in [0.2, 0.4, 0.6, 0.8, 1.0]:
            n_rows = int(len(X_train) * fraction)
            temp_model = model.__class__(**model.get_params())
            temp_model.fit(X_train.iloc[:n_rows], y_train.iloc[:n_rows])
            temp_pred = temp_model.predict(X_test)
            loss_values.append(rmse(y_test, temp_pred))

        loss_curve_path = artifact_dir / "loss_curve.png"
        save_loss_curve(model_name, loss_values, str(loss_curve_path))
        mlflow.log_artifact(str(loss_curve_path))

        prediction_plot_path = artifact_dir / "actual_vs_predicted.png"
        save_prediction_plot(y_test, predictions, str(prediction_plot_path))
        mlflow.log_artifact(str(prediction_plot_path))

        metrics_path = artifact_dir / "metrics.json"
        with open(metrics_path, "w", encoding="utf-8") as file:
            json.dump(metrics, file, indent=4)
        mlflow.log_artifact(str(metrics_path))

        mlflow.sklearn.log_model(sk_model=model, artifact_path="model")

        return {
            "run_id": run.info.run_id,
            "model_name": model_name,
            "metrics": metrics,
            "model": model,
        }


def run_optuna_hpo_for_algorithm(
    algorithm: str,
    X_train,
    y_train,
    X_test,
    y_test,
    params: dict,
) -> dict:
    random_state = params["project"]["random_state"]
    n_trials = params["training"]["hpo"]["n_trials"]

    study = optuna.create_study(direction="minimize")

    study.optimize(
        lambda trial: objective(
            trial,
            algorithm,
            X_train,
            y_train,
            random_state,
        ),
        n_trials=n_trials,
    )

    best_model = create_model_from_trial(
        study.best_trial,
        algorithm,
        random_state,
    )

    run_name = f"optuna_{algorithm}"
    artifact_dir = Path("artifacts") / run_name
    artifact_dir.mkdir(parents=True, exist_ok=True)

    start_time = time.time()

    with mlflow.start_run(run_name=run_name) as run:
        best_model.fit(X_train, y_train)

        training_time = time.time() - start_time
        predictions = best_model.predict(X_test)
        metrics = evaluate_model(best_model, X_test, y_test)

        mlflow.log_param("algorithm", algorithm)
        mlflow.log_param("run_type", "hpo")
        mlflow.log_param("hpo_method", "optuna")
        mlflow.log_param("n_trials", n_trials)
        mlflow.log_params(study.best_params)
        log_dataset_metadata(X_train, X_test)

        mlflow.log_metric("training_time_seconds", training_time)
        mlflow.log_metric("best_cv_rmse", float(study.best_value))

        for metric_name, value in metrics.items():
            mlflow.log_metric(f"test_{metric_name}", value)

        trial_values = [trial.value for trial in study.trials if trial.value is not None]

        loss_curve_path = artifact_dir / "optuna_loss_curve.png"
        save_loss_curve(run_name, trial_values, str(loss_curve_path))
        mlflow.log_artifact(str(loss_curve_path))

        prediction_plot_path = artifact_dir / "actual_vs_predicted.png"
        save_prediction_plot(y_test, predictions, str(prediction_plot_path))
        mlflow.log_artifact(str(prediction_plot_path))

        metrics_path = artifact_dir / "metrics.json"
        with open(metrics_path, "w", encoding="utf-8") as file:
            json.dump(metrics, file, indent=4)
        mlflow.log_artifact(str(metrics_path))

        best_params_path = artifact_dir / "best_params.json"
        with open(best_params_path, "w", encoding="utf-8") as file:
            json.dump(study.best_params, file, indent=4)
        mlflow.log_artifact(str(best_params_path))

        mlflow.sklearn.log_model(sk_model=best_model, artifact_path="model")

        return {
            "run_id": run.info.run_id,
            "model_name": run_name,
            "metrics": metrics,
            "model": best_model,
        }


def export_experiment_log(experiment_name: str) -> None:
    experiment = mlflow.get_experiment_by_name(experiment_name)

    if experiment is None:
        raise ValueError(f"Experiment not found: {experiment_name}")

    runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])

    os.makedirs("docs", exist_ok=True)
    runs.to_csv("docs/experiment_log.csv", index=False)


def save_best_model(best_result: dict) -> None:
    os.makedirs("models", exist_ok=True)

    joblib.dump(best_result["model"], "models/model.joblib")

    with open("models/best_run.json", "w", encoding="utf-8") as file:
        json.dump(
            {
                "run_id": best_result["run_id"],
                "model_name": best_result["model_name"],
                "metrics": best_result["metrics"],
            },
            file,
            indent=4,
        )


def main() -> None:
    params = load_params()

    mlflow.set_tracking_uri(params["mlflow"]["tracking_uri"])
    mlflow.set_experiment(params["training"]["experiment_name"])

    random_state = params["project"]["random_state"]
    algorithms = params["training"]["algorithms"]

    X_train, y_train, X_test, y_test = load_data()

    results = []

    baseline_models = get_baseline_models(random_state)

    for model_name, model in baseline_models.items():
        if model_name in algorithms:
            result = train_and_log_baseline_model(
                model_name,
                model,
                X_train,
                y_train,
                X_test,
                y_test,
            )
            results.append(result)

    for algorithm in algorithms:
        hpo_result = run_optuna_hpo_for_algorithm(
            algorithm,
            X_train,
            y_train,
            X_test,
            y_test,
            params,
        )
        results.append(hpo_result)

    best_result = min(results, key=lambda item: item["metrics"]["rmse"])

    save_best_model(best_result)
    export_experiment_log(params["training"]["experiment_name"])

    print("Training completed.")
    print(f"Best run ID: {best_result['run_id']}")
    print(f"Best model: {best_result['model_name']}")
    print(f"Best RMSE: {best_result['metrics']['rmse']}")


if __name__ == "__main__":
    main()