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

import os
import time
from typing import Dict, List

import requests
from pyspark.dbutils import DBUtils
from pyspark.sql import SparkSession

from tennis.config import ProjectConfig
from tennis.serving.model_serving import ModelServing

# spark session

spark = SparkSession.builder.getOrCreate()
dbutils = DBUtils(spark)

# get environment variables
os.environ["DBR_TOKEN"] = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
os.environ["DBR_HOST"] = spark.conf.get("spark.databricks.workspaceUrl")

# Load project config
config = ProjectConfig.from_yaml(config_path="../project_config.yml")
catalog_name = config.catalog_name
schema_name = config.schema_name

# COMMAND ----------

# Initialize feature store manager
model_serving = ModelServing(
    model_name=f"{catalog_name}.{schema_name}.pyfunc-tennis-model", endpoint_name="tennis-model-serving"
)

# COMMAND ----------

# Deploy the model serving endpoint
model_serving.deploy_or_update_serving_endpoint()


# COMMAND ----------

# Create a sample request body

# Sample 1000 records from the training set
test_set = spark.table(f"{config.catalog_name}.{config.schema_name}.test_set").toPandas()
test_set = test_set.drop(["update_timestamp_utc", "Id", "RESULT"], axis=1)

# Sample 100 records from the training set
sampled_records = test_set.sample(n=100, replace=True).to_dict(orient="records")
dataframe_records = [[record] for record in sampled_records]

# COMMAND ----------

dataframe_records

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
    serving_endpoint = f"https://{os.environ['DBR_HOST']}/serving-endpoints/tennis-model-serving/invocations"

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
