import json
import time

import mlflow
import yaml
from mlflow.tracking import MlflowClient


def load_params(path: str = "configs/params.yaml") -> dict:
    with open(path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def load_best_run(path: str = "models/best_run.json") -> dict:
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def wait_for_model_version(client, model_name: str, version: str) -> None:
    for _ in range(30):
        model_version = client.get_model_version(model_name, version)
        if model_version.status == "READY":
            return
        time.sleep(1)

    raise TimeoutError(f"Model version {version} was not ready in time.")


def main() -> None:
    params = load_params()
    best_run = load_best_run()

    tracking_uri = params["mlflow"]["tracking_uri"]
    registered_model_name = params["training"]["model_name"]

    mlflow.set_tracking_uri(tracking_uri)

    client = MlflowClient(tracking_uri=tracking_uri)

    model_uri = f"runs:/{best_run['run_id']}/model"

    registered_model = mlflow.register_model(
        model_uri=model_uri,
        name=registered_model_name,
    )

    version = registered_model.version

    wait_for_model_version(client, registered_model_name, version)

    client.transition_model_version_stage(
        name=registered_model_name,
        version=version,
        stage="Staging",
        archive_existing_versions=False,
    )

    print(f"Model {registered_model_name} version {version} moved to Staging.")

    client.transition_model_version_stage(
        name=registered_model_name,
        version=version,
        stage="Production",
        archive_existing_versions=True,
    )

    print(f"Model {registered_model_name} version {version} moved to Production.")


if __name__ == "__main__":
    main()