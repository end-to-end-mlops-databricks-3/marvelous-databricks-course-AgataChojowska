"""Custom model implementation.

infer_signature (from mlflow.models) → Captures input-output schema for model tracking.

num_features → List of numerical feature names.
cat_features → List of categorical feature names.
target → The column to predict.
parameters → Hyperparameters for XGBoost.
catalog_name, schema_name → Database schema names for Databricks tables.
"""

from typing import Optional

import mlflow
import numpy as np
import pandas as pd
from loguru import logger
from mlflow import MlflowClient
from mlflow.data.dataset_source import DatasetSource
from mlflow.models import infer_signature
from mlflow.utils.environment import _mlflow_conda_env
from pyspark.sql import DataFrame as SparkDataFrame
from pyspark.sql import SparkSession
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from tennisprediction.config import ProjectConfig, Tags


def adjust_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    """Make predictions more human readable."""
    return np.round(predictions * 100) / 100


class TennisModelWrapper(mlflow.pyfunc.PythonModel):
    """Wrapper class for machine learning models to be used with MLflow.

    This class wraps a machine learning model for predicting tennis matches winners.
    """

    def __init__(self, model: object) -> None:
        """Initialize the TennisModelWrapper.

        :param model: The underlying machine learning model.
        """
        self.model = model

    def predict(self, context: None, model_input: pd.DataFrame) -> np.array:  # noqua
        """Make predictions using the wrapped model.

        :param context: The MLflow context (unused in this implementation).
        :param model_input: Input data for making predictions.
        :return: A dictionary containing the adjusted prediction.
        """
        logger.info(f"model_input:{model_input}")
        predictions = self.model.predict_proba(model_input)  # try to change here
        logger.info(f"predictions: {predictions}")
        adjusted_predictions = adjust_predictions(predictions)
        logger.info(f"adjusted_predictions: {adjusted_predictions}")
        return adjusted_predictions


class TennisModel:
    """Custom model class for tennis matches prediction.

    This class encapsulates the entire workflow of scaling the data,
    training the model, and making predictions.
    """

    def __init__(
        self,
        config: ProjectConfig,
        spark: SparkSession,
        tags: Tags,
        train_set: pd.DataFrame,
        test_set: pd.DataFrame,
        model_name: Optional[str],
        code_paths: Optional[list] = None,
        additional_pip_deps: Optional[list] = None,
    ) -> None:
        self.config = config
        self.spark = spark
        self.tags = tags.model_dump()
        self.experiment_name = self.config.experiment_name_custom
        self.code_paths = code_paths
        self.X_train = train_set[config.features]
        self.y_train = train_set[config.target_name]
        self.X_test = test_set[config.features]
        self.y_test = test_set[config.target_name]
        self.parameters = self.config.parameters
        self.additional_pip_deps = additional_pip_deps
        self.model_name = model_name or "pyfunc-tennis-model"

    def prepare_features(self) -> None:
        """Prepare features for model training.

        This method sets up a preprocessing pipeline including statistical calculations of
        features and XGBoost regression model.
        """
        logger.info("🔄 Defining preprocessing pipeline...")
        self.pipeline = Pipeline(steps=[("scaler", StandardScaler()), ("classifier", XGBClassifier(**self.parameters))])

        logger.info("✅ Preprocessing pipeline defined.")

    def train(self) -> None:
        """Train the model using the prepared pipeline."""
        logger.info("🚀 Starting training...")
        self.pipeline.fit(self.X_train, self.y_train)

    def log_model(self) -> None:
        """Log the trained model and its metrics to MLflow.

        This method evaluates the model, logs parameters and metrics, and saves the model in MLflow.
        """
        mlflow.set_experiment(self.experiment_name)

        with mlflow.start_run(tags=self.tags) as run:
            self.run_id = run.info.run_id
            predictions = self.pipeline.predict(
                self.X_test
            )  # Predictions on test data that I'll later compare to y_test

            # Evaluate metrics
            accuracy = accuracy_score(self.y_test, predictions)
            logger.info(f"📊 Accuracy: {accuracy}")

            # Log metrics
            mlflow.log_metric("accuracy", accuracy)

            # Log parameters
            mlflow.log_params(self.parameters)

            # Log the model
            signature = infer_signature(model_input=self.X_train, model_output=self.pipeline.predict(self.X_train))

            if isinstance(self.X_train, pd.DataFrame):
                dataset = mlflow.data.from_pandas(
                    self.X_train,
                    name="train_set",
                )
            elif isinstance(self.X_train, SparkDataFrame):
                dataset = mlflow.data.from_spark(
                    self.X_train,
                    table_name=f"{self.catalog_name}.{self.schema_name}.train_set",
                    version=self.data_version,
                )
            else:
                raise ValueError("Unsupported dataset type.")

            mlflow.log_input(dataset, context="training")

            conda_env = _mlflow_conda_env(additional_pip_deps=self.additional_pip_deps)

            mlflow.pyfunc.log_model(
                python_model=TennisModelWrapper(self.pipeline),
                artifact_path=self.model_name,
                code_paths=self.code_paths,
                conda_env=conda_env,
                signature=signature,
                input_example=self.X_train.iloc[0:1],
            )

    def register_model(self) -> None:
        """Register the trained model in MLflow Model Registry.

        This method registers the model and sets an alias for the latest version.
        """
        logger.info("🔄 Registering the model in UC...")
        self.model_uri = f"runs:/{self.run_id}/{self.model_name}"
        registered_model = mlflow.register_model(
            model_uri=self.model_uri,
            name=f"{self.config.catalog_name}.{self.config.schema_name}.{self.model_name}",
            tags=self.tags,
        )
        logger.info(f"✅ Model registered as version {registered_model.version}.")

        latest_version = registered_model.version

        client = MlflowClient()
        client.set_registered_model_alias(
            name=f"{self.config.catalog_name}.{self.config.schema_name}.{self.model_name}",
            alias="latest-model",
            version=latest_version,
        )

    def retrieve_current_run_dataset(self) -> DatasetSource:
        """Retrieve the dataset used in the current MLflow run.

        :return: The loaded dataset source.
        """
        run = mlflow.get_run(self.run_id)
        dataset_info = run.inputs.dataset_inputs[0].dataset
        dataset_source = mlflow.data.get_source(dataset_info)
        logger.info("✅ Dataset source loaded.")
        return dataset_source.load()

    def retrieve_current_run_metadata(self) -> tuple[dict, dict]:
        """Retrieve metadata from the current MLflow run.

        :return: A tuple containing metrics and parameters of the current run.
        """
        run = mlflow.get_run(self.run_id)
        metrics = run.data.to_dictionary()["metrics"]
        params = run.data.to_dictionary()["params"]
        logger.info("✅ Dataset metadata loaded.")
        return metrics, params

    def load_latest_model_and_predict(self, inference_data: pd.DataFrame) -> np.ndarray:
        """Load the latest model (alias=latest-model) from MLflow and make predictions.

        Alias latest is not allowed -> we use latest-model instead as an alternative.

        :param inference_data: Input data for prediction.
        :return: Predictions.

        Note:
        This also works
        model.unwrap_python_model().predict(None, inference_data)
        check out this article:
        https://medium.com/towards-data-science/algorithm-agnostic-model-building-with-mlflow-b106a5a29535

        """
        logger.info("🔄 Loading model from MLflow alias 'production'...")

        model_uri = f"models:/{self.config.catalog_name}.{self.config.schema_name}.{self.model_name}@latest-model"
        model = mlflow.pyfunc.load_model(model_uri)

        logger.info("✅ Model successfully loaded.")

        # Make predictions: None is context
        probs = model.predict(inference_data)
        prob_player1_wins = probs[0][1]
        prob_player2_wins = probs[0][0]

        # Return predictions as a DataFrame
        return prob_player1_wins, prob_player2_wins
