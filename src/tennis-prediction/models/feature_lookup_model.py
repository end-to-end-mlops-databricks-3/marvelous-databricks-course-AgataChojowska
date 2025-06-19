"""FeatureLookUp model implementation."""

import mlflow
from databricks import feature_engineering
from databricks.feature_engineering import FeatureFunction, FeatureLookup
from databricks.sdk import WorkspaceClient
from loguru import logger
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient
from pyspark.sql import DataFrame, SparkSession
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from tennis.config import ProjectConfig, Tags


class FeatureLookUpModel:
    """A class to manage FeatureLookupModel."""

    def __init__(self, config: ProjectConfig, tags: Tags, spark: SparkSession) -> None:
        """Initialize the model with project configuration."""
        self.config = config
        self.spark = spark
        self.workspace = WorkspaceClient()
        self.fe = feature_engineering.FeatureEngineeringClient()

        # Extract settings from the config
        self.features = self.config.features
        self.target = self.config.target_name
        self.parameters = self.config.parameters
        self.catalog_name = self.config.catalog_name
        self.schema_name = self.config.schema_name

        # Define table names and function name
        self.feature_table_name = f"{self.catalog_name}.{self.schema_name}.tennis_features"
        self.function_name = f"{self.catalog_name}.{self.schema_name}.age_diff_in_months"

        # MLflow configuration
        self.experiment_name = self.config.experiment_name_fe
        self.tags = tags.dict()

    def create_feature_table(self) -> None:
        """Create or update the house_features table and populate it.

        This table stores features related to houses.
        """
        self.spark.sql(f"""
        CREATE OR REPLACE TABLE {self.feature_table_name}
        (Id STRING NOT NULL, AGE_DIFF INT, DRAW_SIZE INT, ATP_POINTS_DIFF INT);
        """)
        self.spark.sql(f"ALTER TABLE {self.feature_table_name} ADD CONSTRAINT house_pk PRIMARY KEY(Id);")
        self.spark.sql(f"ALTER TABLE {self.feature_table_name} SET TBLPROPERTIES (delta.enableChangeDataFeed = true);")

        self.spark.sql(
            f"INSERT INTO {self.feature_table_name} SELECT Id, AGE_DIFF, DRAW_SIZE, ATP_POINTS_DIFF FROM {self.catalog_name}.{self.schema_name}.train_set"
        )
        self.spark.sql(
            f"INSERT INTO {self.feature_table_name} SELECT Id, AGE_DIFF, DRAW_SIZE, ATP_POINTS_DIFF FROM {self.catalog_name}.{self.schema_name}.test_set"
        )
        logger.info("✅ Feature table created and populated.")

    def define_feature_function(self) -> None:
        """Define a function to calculate the house's age.

        This function subtracts the year built from the current year.
        """
        self.spark.sql(f"""
        CREATE OR REPLACE FUNCTION {self.function_name}(age_diff INT)
        RETURNS INT
        LANGUAGE PYTHON AS
        $$
        return age_diff * 12
        $$
        """)
        logger.info("✅ Feature function defined.")

    def load_data(self) -> None:
        """Load training and testing data from Delta tables.

        Drops specified columns and casts 'YearBuilt' to integer type.
        """
        self.train_set = self.spark.table(f"{self.catalog_name}.{self.schema_name}.train_set").drop(
            "AGE_DIFF", "DRAW_SIZE", "ATP_POINTS_DIFF"
        )
        self.test_set = self.spark.table(f"{self.catalog_name}.{self.schema_name}.test_set").toPandas()

        self.train_set = self.train_set.withColumn("Id", self.train_set["Id"].cast("string"))

        logger.info("✅ Data successfully loaded.")

    def feature_engineering(self) -> None:
        """Perform feature engineering by linking data with feature tables.

        Creates a training set using FeatureLookup and FeatureFunction.
        """
        self.training_set = self.fe.create_training_set(
            df=self.train_set,
            label=self.target,
            feature_lookups=[
                FeatureLookup(
                    table_name=self.feature_table_name,
                    feature_names=["AGE_DIFF", "DRAW_SIZE", "ATP_POINTS_DIFF"],
                    lookup_key="Id",
                ),
                FeatureFunction(
                    udf_name=self.function_name,
                    output_name="age_diff_in_months",
                    input_bindings={"age_diff": "AGE_DIFF"},
                ),
            ],
            exclude_columns=["update_timestamp_utc"],
        )

        self.training_df = self.training_set.load_df().toPandas()

        self.test_set["age_diff_in_months"] = (
            self.test_set["AGE_DIFF"] * 12
        )  # TODO  Why are we doing it here? Shouldn't FE be responsible for adding that because we defined the function above?

        self.X_train = self.training_df[self.features + ["age_diff_in_months"]]
        self.y_train = self.training_df[self.target]
        self.X_test = self.test_set[self.features + ["age_diff_in_months"]]
        self.y_test = self.test_set[self.target]

        logger.info("✅ Feature engineering completed.")

    def train(self) -> None:
        """Train the model and log results to MLflow.

        Uses a pipeline with preprocessing and XGBoost regressor.
        """
        logger.info("🚀 Starting training...")

        pipeline = Pipeline(steps=[("scaler", StandardScaler()), ("classifier", XGBClassifier(**self.parameters))])

        mlflow.set_experiment(self.experiment_name)

        with mlflow.start_run(tags=self.tags) as run:
            self.run_id = run.info.run_id
            pipeline.fit(self.X_train, self.y_train)
            y_pred = pipeline.predict(self.X_test)

            accuracy = accuracy_score(self.y_test, y_pred)
            logger.info(f"📊 Accuracy: {accuracy}")

            mlflow.log_param("model_type", "XGBoost with scaling")
            mlflow.log_params(self.parameters)
            mlflow.log_metric("accuracy", accuracy)
            signature = infer_signature(self.X_train, y_pred)

            self.fe.log_model(
                model=pipeline,
                flavor=mlflow.sklearn,
                artifact_path="xgboost-pipeline-model-fe",
                training_set=self.training_set,
                signature=signature,
            )

    def register_model(self) -> str:
        """Register the trained model to MLflow registry.

        Registers the model and sets alias to 'latest-model'.
        """
        registered_model = mlflow.register_model(
            model_uri=f"runs:/{self.run_id}/xgboost-pipeline-model-fe",
            name=f"{self.catalog_name}.{self.schema_name}.tennis_model_fe",
            tags=self.tags,
        )

        # Fetch the latest version dynamically
        latest_version = registered_model.version

        client = MlflowClient()
        client.set_registered_model_alias(
            name=f"{self.catalog_name}.{self.schema_name}.tennis_model_fe",
            alias="latest-model",
            version=latest_version,
        )

        return latest_version

    def load_latest_model_and_predict(self, X: DataFrame) -> DataFrame:
        """Load the trained model from MLflow using Feature Engineering Client and make predictions.

        Loads the model with the alias 'latest-model' and scores the batch.
        :param X: DataFrame containing the input features.
        :return: DataFrame containing the predictions.
        """
        model_uri = f"models:/{self.catalog_name}.{self.schema_name}.tennis_model_fe@latest-model"

        predictions = self.fe.score_batch(model_uri=model_uri, df=X)
        return predictions
