"""Unit tests for TennisModel."""

import mlflow
import pandas as pd
from conftest import TRACKING_URI
from loguru import logger
from mlflow.entities.model_registry.registered_model import RegisteredModel
from mlflow.tracking import MlflowClient
from pyspark.sql import SparkSession
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from tennisprediction.config import ProjectConfig, Tags
from tennisprediction.models.custom_model import TennisModel

mlflow.set_tracking_uri(TRACKING_URI)


def test_custom_model_init(
    config: ProjectConfig, tags: Tags, spark_session: SparkSession, train_set: pd.DataFrame, test_set: pd.DataFrame
) -> None:
    """Test the initialization of TennisModel.

    This function creates a TennisModel instance and asserts that its attributes are of the correct types.

    :param config: Configuration for the project
    :param tags: Tags associated with the model
    :param spark_session: Spark session object
    """
    model = TennisModel(
        config=config, tags=tags, spark=spark_session, code_paths=[], train_set=train_set, test_set=test_set
    )
    assert isinstance(model, TennisModel)
    assert isinstance(model.config, ProjectConfig)
    assert isinstance(model.tags, dict)
    assert isinstance(model.spark, SparkSession)
    assert isinstance(model.code_paths, list)
    assert not model.code_paths


def test_prepare_features(mock_custom_model: TennisModel) -> None:
    """Test that prepare_features method initializes pipeline components correctly.

    Verifies the preprocessor is a ColumnTransformer and pipeline contains expected
    ColumnTransformer and LGBMRegressor steps in sequence.

    :param mock_custom_model: Mocked TennisModel instance for testing
    """
    mock_custom_model.prepare_features()
    assert isinstance(mock_custom_model.pipeline, Pipeline)
    assert isinstance(mock_custom_model.pipeline.steps, list)
    assert isinstance(mock_custom_model.pipeline.steps[0][1], StandardScaler)
    assert isinstance(mock_custom_model.pipeline.steps[1][1], XGBClassifier)


def test_train(mock_custom_model: TennisModel) -> None:
    """Test that train method configures pipeline with correct feature handling.

    Validates feature count matches configuration and feature names align with
    numerical/categorical features defined in model config.

    :param mock_custom_model: Mocked TennisModel instance for testing
    """
    mock_custom_model.prepare_features()
    mock_custom_model.train()

    assert mock_custom_model.pipeline.n_features_in_ == len(mock_custom_model.config.features)
    assert sorted(mock_custom_model.config.features) == sorted(mock_custom_model.pipeline.feature_names_in_)


def test_log_model_with_PandasDataset(mock_custom_model: TennisModel) -> None:
    """Test model logging with PandasDataset validation.

    Verifies that the model's pipeline captures correct feature dimensions and names,
    then checks proper dataset type handling during model logging.

    :param mock_custom_model: Mocked TennisModel instance for testing
    """
    mock_custom_model.prepare_features()
    mock_custom_model.train()

    assert mock_custom_model.pipeline.n_features_in_ == len(mock_custom_model.config.features)
    assert sorted(mock_custom_model.config.features) == sorted(mock_custom_model.pipeline.feature_names_in_)

    mock_custom_model.log_model()

    # Split the following part
    client = MlflowClient()
    experiment = mlflow.get_experiment_by_name(mock_custom_model.experiment_name)
    assert experiment.name == mock_custom_model.experiment_name

    experiment_id = experiment.experiment_id
    assert experiment_id

    runs = client.search_runs(experiment_id, order_by=["start_time desc"], max_results=1)
    assert len(runs) == 1
    latest_run = runs[0]

    model_uri = f"runs:/{latest_run.info.run_id}/model"
    logger.info(f"{model_uri= }")

    assert model_uri


def test_register_model(mock_custom_model: TennisModel) -> None:
    """Test the registration of a custom MLflow model.

    This function performs several operations on the mock custom model, including loading data,
    preparing features, training, and logging the model. It then registers the model and verifies
    its existence in the MLflow model registry.

    :param mock_custom_model: A mocked instance of the TennisModel class.
    """
    mock_custom_model.prepare_features()
    mock_custom_model.train()
    mock_custom_model.log_model()

    mock_custom_model.register_model()

    client = MlflowClient()
    model_name = f"{mock_custom_model.config.catalog_name}.{mock_custom_model.config.schema_name}.pyfunc-tennis-model"

    try:
        model = client.get_registered_model(model_name)
        logger.info(f"Model '{model_name}' is registered.")
        logger.info(f"Latest version: {model.latest_versions[-1].version}")
        logger.info(f"{model.name = }")
    except mlflow.exceptions.MlflowException as e:
        if "RESOURCE_DOES_NOT_EXIST" in str(e):
            logger.error(f"Model '{model_name}' is not registered.")
        else:
            raise e

    assert isinstance(model, RegisteredModel)
    alias, version = model.aliases.popitem()
    assert alias == "latest-model"


def test_retrieve_current_run_metadata(mock_custom_model: TennisModel) -> None:
    """Test retrieving the current run metadata from a mock custom model.

    This function verifies that the `retrieve_current_run_metadata` method
    of the `TennisModel` class returns metrics and parameters as dictionaries.

    :param mock_custom_model: A mocked instance of the TennisModel class.
    """
    mock_custom_model.prepare_features()
    mock_custom_model.train()
    mock_custom_model.log_model()

    metrics, params = mock_custom_model.retrieve_current_run_metadata()
    assert isinstance(metrics, dict)
    assert metrics
    assert isinstance(params, dict)
    assert params


def test_load_latest_model_and_predict(mock_custom_model: TennisModel) -> None:
    """Test the process of loading the latest model and making predictions.

    This function performs the following steps:
    - Loads data using the provided custom model.
    - Prepares features and trains the model.
    - Logs and registers the trained model.
    - Extracts input data from the test set and makes predictions using the latest model.

    :param mock_custom_model: Instance of a custom machine learning model with methods for data
                              loading, feature preparation, training, logging, and prediction.
    """
    mock_custom_model.prepare_features()
    mock_custom_model.train()
    mock_custom_model.log_model()
    mock_custom_model.register_model()

    input_data = mock_custom_model.X_test
    input_data = input_data.where(input_data.notna(), None)  # noqa

    for row in input_data.itertuples(index=False):
        row_df = pd.DataFrame([row._asdict()])
        print(row_df.to_dict(orient="split"))
        predictions = mock_custom_model.load_latest_model_and_predict(inference_data=row_df)

        assert len(predictions) == 2
