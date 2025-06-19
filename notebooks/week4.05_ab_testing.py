# Databricks notebook source
# MAGIC %pip install -e ..
# MAGIC %pip install git+https://github.com/end-to-end-mlops-databricks-3/marvelous@0.1.0

# COMMAND ----------

# MAGIC %restart_python

# COMMAND ----------

from pathlib import Path
import sys
sys.path.append(str(Path.cwd().parent / 'src'))

# COMMAND ----------

import hashlib

import mlflow
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import (
    EndpointCoreConfigInput,
    ServedEntityInput,
)
from mlflow.models import infer_signature
from pyspark.sql import SparkSession

from tennisprediction.config import ProjectConfig, Tags
from tennisprediction.models.custom_model import TennisModel
from marvelous.common import is_databricks
from dotenv import load_dotenv
import os
import requests
import time

# COMMAND ----------

if not is_databricks():
    load_dotenv()
    profile = os.environ["PROFILE"]
    mlflow.set_tracking_uri(f"databricks://{profile}")
    mlflow.set_registry_uri(f"databricks-uc://{profile}")


config = ProjectConfig.from_yaml(config_path="../project_config.yml", env="prd")
spark = SparkSession.builder.getOrCreate()
tags = Tags(**{"git_sha": "abcd12345", "branch": "week4", "job_run_id": "1"})

# COMMAND ----------

# Load project config
config = ProjectConfig.from_yaml(config_path="../project_config.yml", env="dev")
catalog_name = config.catalog_name
schema_name = config.schema_name

# Load test and train data
from tennisprediction.catalog_utils import load_from_table_to_pandas

train_set = load_from_table_to_pandas(spark=spark, config=config, table="train_set").drop(
        ["update_timestamp_utc"], axis=1
    )
test_set = load_from_table_to_pandas(spark=spark, config=config, table="test_set").drop(
    ["update_timestamp_utc"], axis=1
)


# COMMAND ----------

# train model A

basic_model = TennisModel(config=config, tags=tags, spark=spark, train_set=train_set, test_set=test_set, 
                          model_name="tennis_model_A", additional_pip_deps = ["./code/tennisprediction-0.0.1-py3-none-any.whl"], 
                          code_paths=["../dist/tennisprediction-0.0.1-py3-none-any.whl"]) # This wheel needs to be created for example with uv build.

basic_model.prepare_features()
basic_model.train()
basic_model.log_model()
basic_model.register_model()
model_A_uri = basic_model.model_uri
print(model_A_uri)

# COMMAND ----------

# train model B
basic_model_b = TennisModel(config=config, tags=tags, spark=spark, train_set=train_set, test_set=test_set, model_name="tennis_model_B")
basic_model_b.paramaters = {"colsample_bytree": 0.8,
                            "learning_rate": 0.07,
                            "max_depth": 7,
                            "n_estimators": 200,
                            "reg_alpha": 0.7,
                            "reg_lambda": 0.7,
                            "subsample": 0.9}
model_name_b = f"{catalog_name}.{schema_name}.tennis_model_B"
basic_model_b.prepare_features()
basic_model_b.train()
basic_model_b.log_model()
basic_model_b.register_model()
model_B_uri = basic_model_b.model_uri
print(model_B_uri)

# COMMAND ----------

# define wrapper
class HousePriceModelWrapper(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        self.model_a = mlflow.pyfunc.load_model(
            context.artifacts["xgboost-pipeline-model-A"]
        )
        self.model_b = mlflow.pyfunc.load_model(
            context.artifacts["xgboost-pipeline-model-B"]
        )

    def predict(self, context, model_input):
        tennis_id = str(model_input["Id"].values[0])
        hashed_id = hashlib.md5(tennis_id.encode(encoding="UTF-8")).hexdigest()
        # convert a hexadecimal (base-16) string into an integer
        if int(hashed_id, 16) % 2:
            predictions = self.model_a.predict(model_input.drop(["Id"], axis=1))
            return {"Prediction": predictions[0], "model": "Model A"}
        else:
            predictions = self.model_b.predict(model_input.drop(["Id"], axis=1))
            return {"Prediction": predictions[0], "model": "Model B"}

# COMMAND ----------

train_set_spark = spark.table(f"{catalog_name}.{schema_name}.train_set")
train_set = train_set_spark.toPandas()
test_set = spark.table(f"{catalog_name}.{schema_name}.test_set").toPandas()
X_train = train_set[config.features + ["Id"]]
X_test = test_set[config.features + ["Id"]]

# COMMAND ----------

conda_env = {
    "name": "mlflow-env",
    "channels": ["conda-forge"],
    "dependencies": [
        "python=3.11.11",
        "pip<=23.2.1",
        {
            "pip": [
                "mlflow==2.17.0",
                "cloudpickle==3.1.0",
                "ipython==8.15.0",
                "numpy==1.26.4", 
                "pandas==2.2.3",
                "pyspark==3.5.0",
                "scikit-learn==1.5.2",
                "scipy==1.14.1",
                "./code/tennisprediction-0.0.1-py3-none-any.whl"
            ]
        }
    ]
}

# COMMAND ----------

mlflow.set_experiment(experiment_name="/Shared/tennis-ab-testing")
model_name = f"{catalog_name}.{schema_name}.tennis_model_pyfunc_ab_test"
wrapped_model = HousePriceModelWrapper()

with mlflow.start_run() as run:
    run_id = run.info.run_id
    signature = infer_signature(model_input=X_train, model_output={"probability1": 14.5, "probability2": 85.5, "model": "Model B"})
    dataset = mlflow.data.from_spark(train_set_spark, table_name=f"{catalog_name}.{schema_name}.train_set", version="0")
    mlflow.log_input(dataset, context="training")
    mlflow.pyfunc.log_model(
        python_model=wrapped_model,
        artifact_path="pyfunc-tennis-model-ab",
        artifacts={
            "xgboost-pipeline-model-A": model_A_uri,
            "xgboost-pipeline-model-B": model_B_uri},
        signature=signature,
        conda_env=conda_env,
        code_paths=["../dist/tennisprediction-0.0.1-py3-none-any.whl"]
    )
model_version = mlflow.register_model(
    model_uri=f"runs:/{run_id}/pyfunc-tennis-model-ab", name=model_name, tags=tags.dict()
)

# COMMAND ----------

"""Model serving module."""

workspace = WorkspaceClient()
model_name=f"{catalog_name}.{schema_name}.tennis_model_pyfunc_ab_test"
endpoint_name="tennis-custom-model-serving-ab"
entity_version = model_version.version # registered model version

endpoint_exists = any(item.name == endpoint_name for item in workspace.serving_endpoints.list())

# get environment variables
os.environ["DBR_TOKEN"] = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
os.environ["DBR_HOST"] = spark.conf.get("spark.databricks.workspaceUrl")

served_entities = [
    ServedEntityInput(
        entity_name=model_name,
        scale_to_zero_enabled="True",
        workload_size="Small",
        entity_version=entity_version,
    )
]

if not endpoint_exists:
    workspace.serving_endpoints.create(
        name=endpoint_name,
        config=EndpointCoreConfigInput(
            served_entities=served_entities,
        ),
    )
else:
    workspace.serving_endpoints.update_config(name=endpoint_name, served_entities=served_entities)

# COMMAND ----------

# Sample 1000 records from the test set
test_set = spark.table(f"{config.catalog_name}.{config.schema_name}.test_set").toPandas()
test_set = test_set.drop(["update_timestamp_utc", "RESULT"], axis=1)
sampled_records = test_set.sample(n=1000, replace=True).to_dict(orient="records")
dataframe_records = [[record] for record in sampled_records]
print(dataframe_records[0])


# COMMAND ----------

# Call the endpoint with one sample record

"""
Each dataframe record in the request body should be list of json with columns looking like:

[{'LotFrontage': 78.0,
  'LotArea': 9317,
  'OverallQual': 6,
  'OverallCond': 5,
  'YearBuilt': 2006,
  'Exterior1st': 'VinylSd',
  'Exterior2nd': 'VinylSd',
  'MasVnrType': 'None',
  'Foundation': 'PConc',
  'Heating': 'GasA',
  'CentralAir': 'Y',
  'SaleType': 'WD',
  'SaleCondition': 'Normal'}]
"""

def call_endpoint(record):
    """
    Calls the model serving endpoint with a given input record.
    """
    serving_endpoint = f"https://{os.environ['DBR_HOST']}/serving-endpoints/tennis-custom-model-serving-ab/invocations"

    response = requests.post(
        serving_endpoint,
        headers={"Authorization": f"Bearer {os.environ['DBR_TOKEN']}"},
        json={"dataframe_records": record},
    )
    return response.status_code, response.text


status_code, response_text = call_endpoint(dataframe_records[0])
print(f"Response Status: {status_code}")
print(f"Response Text: {response_text}")

# COMMAND ----------

# Load test
for i in range(len(dataframe_records)):
    status_code, response_text = call_endpoint(dataframe_records[i])
    print(f"Response Status: {status_code}")
    print(f"Response Text: {response_text}")
    time.sleep(0.2)
