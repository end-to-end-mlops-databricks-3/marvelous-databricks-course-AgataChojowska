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

# MAGIC %md
# MAGIC ## Feature Importance

# COMMAND ----------

import pandas as pd
from pyspark.sql.functions import col
from pyspark.sql.functions import current_timestamp, to_utc_timestamp
import numpy as np
from pyspark.sql import SparkSession
from loguru import logger
from tennisprediction.config import ProjectConfig

spark = SparkSession.builder.getOrCreate()

# Load configuration
config = ProjectConfig.from_yaml(config_path="../project_config.yml", env="dev")
spark = SparkSession.builder.getOrCreate()

train_set = spark.table(f"{config.catalog_name}.{config.schema_name}.train_set").toPandas()
test_set = spark.table(f"{config.catalog_name}.{config.schema_name}.test_set").toPandas()

# COMMAND ----------


import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

# Encode categorical and datetime variables
def preprocess_data(df):
    label_encoders = {}
    for col in df.select_dtypes(include=['object', 'datetime']).columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le
    return df, label_encoders

train_set, label_encoders = preprocess_data(train_set)

# Define features and target (adjust columns accordingly)
features = train_set.drop(columns=["RESULT"])
target = train_set["RESULT"]

# Train a Random Forest model
model = RandomForestRegressor(random_state=42)
model.fit(features, target)

# Identify the most important features
feature_importances = pd.DataFrame({
    'Feature': features.columns,
    'Importance': model.feature_importances_
}).sort_values(by='Importance', ascending=False)

print("Top 5 important features:")
print(feature_importances.head(5))


# COMMAND ----------

test_set.head()

# COMMAND ----------

part_df = test_set[["Id", "N_GAMES_DIFF", "ATP_POINTS_DIFF", "ELO_DIFF", "ATP_RANK_DIFF"]]
part_df.head(20)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Generate Synthetic Data

# COMMAND ----------

import pandas as pd
import numpy as np
from typing import Optional
from datetime import datetime


def generate_synthetic_data(
    df: pd.DataFrame, 
    drift: bool = False, 
    n_samples: Optional[int] = None,
    drift_factor: float = 1.5,
    random_state: Optional[int] = None
) -> pd.DataFrame:
    """
    Generate synthetic data with option for drift.
    
    Creates new values for specific columns while preserving others.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input DataFrame containing the original data
    drift : bool, default=False
        If True, generates data with distributional drift
    n_samples : int, optional
        Number of synthetic samples to generate. If None, uses len(df)
    drift_factor : float, default=1.5
        Factor to apply drift (only used when drift=True)
    random_state : int, optional
        Random seed for reproducibility
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with synthetic values for specified columns and original values for others
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    if n_samples is None:
        n_samples = len(df)
    
    # Create a copy of the original DataFrame structure
    synthetic_df = df.sample(n=n_samples, replace=True, random_state=random_state).reset_index(drop=True)
    
    # Generate synthetic IDs (year + match format like 2022xxxx)
    current_year = datetime.now().year
    base_ids = [f"{current_year}{str(i).zfill(6)}" for i in range(100000, 100000 + n_samples)]
    synthetic_df['Id'] = base_ids
    
    # Define the columns we want to modify
    target_columns = ['N_GAMES_DIFF', 'ATP_POINTS_DIFF', 'ELO_DIFF', 'ATP_RANK_DIFF']
    
    # Generate synthetic values for each target column
    for col in target_columns:
        if col not in df.columns:
            continue
            
        # Get original distribution statistics
        original_mean = df[col].mean()
        original_std = df[col].std()
        original_min = df[col].min()
        original_max = df[col].max()
        
        if not drift:
            # No drift: generate data with same distribution
            synthetic_values = np.random.normal(original_mean, original_std, n_samples)
        else:
            # With drift: shift the distribution
            if col == 'N_GAMES_DIFF':
                # Simulate drift: players with more game experience difference
                shifted_mean = original_mean * drift_factor
                shifted_std = original_std * 1.2
            elif col == 'ATP_POINTS_DIFF':
                # Simulate drift: larger point differences (more polarized rankings)
                shifted_mean = original_mean * drift_factor
                shifted_std = original_std * drift_factor
            elif col == 'ELO_DIFF':
                # Simulate drift: ELO differences become more extreme
                shifted_mean = original_mean * 1.3
                shifted_std = original_std * drift_factor
            elif col == 'ATP_RANK_DIFF':
                # Simulate drift: rank differences become larger
                shifted_mean = original_mean * drift_factor
                shifted_std = original_std * 1.4
            else:
                # Default drift behavior
                shifted_mean = original_mean * drift_factor
                shifted_std = original_std * drift_factor
            
            synthetic_values = np.random.normal(shifted_mean, shifted_std, n_samples)
        
        # Apply realistic constraints based on the column
        if col == 'N_GAMES_DIFF':
            # Games difference should be integers and reasonable
            synthetic_values = np.round(synthetic_values).astype(int)
            synthetic_values = np.clip(synthetic_values, -500, 500)
        elif col == 'ATP_POINTS_DIFF':
            # Points difference should be reasonable
            synthetic_values = np.round(synthetic_values, 1)
            synthetic_values = np.clip(synthetic_values, -10000, 10000)
        elif col == 'ELO_DIFF':
            # ELO difference should be reasonable
            synthetic_values = np.round(synthetic_values, 6)
            synthetic_values = np.clip(synthetic_values, -500, 500)
        elif col == 'ATP_RANK_DIFF':
            # Rank difference should be reasonable
            synthetic_values = np.round(synthetic_values, 1)
            synthetic_values = np.clip(synthetic_values, -1000, 1000)
        
        # Assign the synthetic values
        synthetic_df[col] = synthetic_values
    
    return synthetic_df


    
    # # Generate synthetic data without drift
    # synthetic_no_drift = generate_synthetic_data(df, drift=False, n_samples=10, random_state=42)
    # print("Synthetic data WITHOUT drift:")
    # print(synthetic_no_drift[['Id', 'N_GAMES_DIFF', 'ATP_POINTS_DIFF', 'ELO_DIFF', 'ATP_RANK_DIFF']])
    # print("\n" + "="*50 + "\n")
    
    # # Generate synthetic data with drift
    # synthetic_with_drift = generate_synthetic_data(df, drift=True, n_samples=10, random_state=42)
    # print("Synthetic data WITH drift:")
    # print(synthetic_with_drift[['Id', 'N_GAMES_DIFF', 'ATP_POINTS_DIFF', 'ELO_DIFF', 'ATP_RANK_DIFF']])
    # print("\n" + "="*50 + "\n")
    

# COMMAND ----------

inference_data_skewed = generate_synthetic_data(train_set, drift= True, n_samples=200, random_state=42, drift_factor=5)

# COMMAND ----------

inference_data_skewed.head()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Create Tables and Update house_features_online

# COMMAND ----------

inference_data_skewed_spark = spark.createDataFrame(inference_data_skewed).withColumn(
    "update_timestamp_utc", to_utc_timestamp(current_timestamp(), "UTC")
)

inference_data_skewed_spark.write.mode("overwrite").saveAsTable(
    f"{config.catalog_name}.{config.schema_name}.inference_data_skewed"
)

# COMMAND ----------

import time
from databricks.sdk import WorkspaceClient
# It actually works but is delayed.

workspace = WorkspaceClient()

#write into feature table; update online table
import time
from databricks.sdk import WorkspaceClient

workspace = WorkspaceClient()

#write into feature table; update online table
spark.sql(f"""
    INSERT INTO {config.catalog_name}.{config.schema_name}.tennis_features
    SELECT Id, AGE_DIFF, DRAW_SIZE, ATP_POINTS_DIFF
    FROM {config.catalog_name}.{config.schema_name}.inference_data_skewed
""")
  
online_table_name = f"{config.catalog_name}.{config.schema_name}.tennis_features_online"

existing_table = workspace.online_tables.get(online_table_name)
logger.info("Online table already exists. Inititating table update.")
pipeline_id = existing_table.spec.pipeline_id
update_response = workspace.pipelines.start_update(pipeline_id=pipeline_id, full_refresh=False)
update_response = workspace.pipelines.start_update(
    pipeline_id=pipeline_id, full_refresh=False)
while True:
    update_info = workspace.pipelines.get_update(pipeline_id=pipeline_id, 
                            update_id=update_response.update_id)
    state = update_info.update.state.value
    if state == 'COMPLETED':
        break
    elif state in ['FAILED', 'CANCELED']:
        raise SystemError("Online table failed to update.")
    elif state == 'WAITING_FOR_RESOURCES':
        print("Pipeline is waiting for resources.")
    else:
        print(f"Pipeline is in {state} state.")
    time.sleep(30)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Send Data to the Endpoint

# COMMAND ----------

import pandas as pd
from pyspark.sql.functions import col
from pyspark.sql.functions import current_timestamp, to_utc_timestamp
import numpy as np
import datetime
import itertools
from pyspark.sql import SparkSession

from tennisprediction.config import ProjectConfig

spark = SparkSession.builder.getOrCreate()

# Load configuration
config = ProjectConfig.from_yaml(config_path="../project_config.yml", env="dev")

test_set = spark.table(f"{config.catalog_name}.{config.schema_name}.test_set") \
                        .withColumn("Id", col("Id").cast("string")) \
                        .toPandas()


inference_data_skewed = spark.table(f"{config.catalog_name}.{config.schema_name}.inference_data_skewed") \
                        .withColumn("Id", col("Id").cast("string")) \
                        .toPandas()


# COMMAND ----------

token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
host = spark.conf.get("spark.databricks.workspaceUrl")

# COMMAND ----------

inference_data_skewed.head()

# COMMAND ----------


from databricks.sdk import WorkspaceClient
import requests
import time

workspace = WorkspaceClient()

sampled_skewed_clean = inference_data_skewed.drop("update_timestamp_utc", axis=1)
test_set_clean = test_set.drop("update_timestamp_utc", axis=1)

# Sample records from inference datasets
sampled_skewed_records = sampled_skewed_clean.sample(n=5, replace=True).to_dict(orient="records")
test_set_records = test_set_clean.sample(n=5, replace=True).to_dict(orient="records")


# COMMAND ----------

# Two different way to send request to the endpoint
# 1. Using https endpoint
def send_request_https(dataframe_record):
    model_serving_endpoint = f"https://{host}/serving-endpoints/pyfunc-simple-endpoint/invocations"
    response = requests.post(
        model_serving_endpoint,
        headers={"Authorization": f"Bearer {token}"},
        json={"dataframe_records": [dataframe_record]},
    )
    return response

# 2. Using workspace client
def send_request_workspace(dataframe_record):
    response = workspace.serving_endpoints.query(
        name="pyfunc-simple-endpoint",
        dataframe_records=[dataframe_record]
    )
    return response

# COMMAND ----------

# The two below "Loop over" blocks are populating the inference table "mlops_dev.chojowsk.tennis-inference-table_payload"

# COMMAND ----------

len(test_set_records)

# COMMAND ----------

import os
import requests
import numpy as np
import pandas as pd
import json

def create_tf_serving_json(data):
    return {'inputs': {name: data[name].tolist() for name in data.keys()} if isinstance(data, dict) else data.tolist()}

def score_model(dataset):
    url = 'https://dbc-c2e8445d-159d.cloud.databricks.com/serving-endpoints/tennis-from-bundle-serving-fe-dev/invocations'
    headers = {'Authorization': f'Bearer {token}', 'Content-Type': 'application/json'}
    ds_dict = {'dataframe_split': dataset.to_dict(orient='split')} if isinstance(dataset, pd.DataFrame) else create_tf_serving_json(dataset)
    data_json = json.dumps(ds_dict, allow_nan=True)
    response = requests.request(method='POST', headers=headers, url=url, data=data_json)
    if response.status_code != 200:
        raise Exception(f'Request failed with status {response.status_code}, {response.text}')
    return response.json()

score_model(test_set_clean)

# COMMAND ----------

# Loop over test records and send requests for 10 minutes
end_time = datetime.datetime.now() + datetime.timedelta(minutes=5)
for index, record in enumerate(itertools.cycle(test_set_records)):
    if datetime.datetime.now() >= end_time:
        break
    print(f"Sending request for test data, index {index}")
    response = send_request_https(record)
    print(f"Response status: {response.status_code}")
    print(f"Response text: {response.text}")
    time.sleep(0.2)


# COMMAND ----------

# end_time = datetime.datetime.now() + datetime.timedelta(minutes=5)
# for index, record in enumerate(itertools.cycle(test_set_records)):
#     if datetime.datetime.now() >= end_time:
#         break
#     print(f"Sending request for test data, index {index}")
#     response = send_request_workspace(record)
#     #print(f"Response status: {response.status_code}")
#     print(f"Response text: {response.text}")
#     time.sleep(0.2)

from tqdm import tqdm
import datetime
import itertools

end_time = datetime.datetime.now() + datetime.timedelta(minutes=5)

# Create progress bar (will auto-update in Databricks)
with tqdm(desc="Processing requests") as pbar:
    for index, record in enumerate(itertools.cycle(test_set_records)):
        if datetime.datetime.now() >= end_time:
            break
        
        try:
            response = send_request_workspace(record)
            status = "✓" if hasattr(response, 'status_code') and response.status_code < 400 else "✓"
            pbar.set_postfix({"Last": f"{status} Index {index}"})
        except Exception as e:
            pbar.set_postfix({"Last": f"✗ Error at {index}"})
        
        pbar.update(1)

# COMMAND ----------

# Loop over skewed records and send requests for 10 minutes
end_time = datetime.datetime.now() + datetime.timedelta(minutes=5)
for index, record in enumerate(itertools.cycle(sampled_skewed_records)):
    if datetime.datetime.now() >= end_time:
        break
    print(f"Sending request for skewed data, index {index}")
    response = send_request_https(record)
    print(f"Response status: {response.status_code}")
    print(f"Response text: {response.text}")
    time.sleep(0.2)

# COMMAND ----------

# Loop over skewed records and send requests for 10 minutes
# end_time = datetime.datetime.now() + datetime.timedelta(minutes=5)
# for index, record in enumerate(itertools.cycle(sampled_skewed_records)):
#     if datetime.datetime.now() >= end_time:
#         break
#     print(f"Sending request for skewed data, index {index}")
#     response = send_request_workspace(record)
#     print(f"Response status: {response.status_code}")
#     print(f"Response text: {response.text}")
#     time.sleep(0.2)

from tqdm import tqdm
import datetime
import itertools

end_time = datetime.datetime.now() + datetime.timedelta(minutes=5)

# Create progress bar (will auto-update in Databricks)
with tqdm(desc="Processing requests") as pbar:
    for index, record in enumerate(itertools.cycle(sampled_skewed_records)):
        if datetime.datetime.now() >= end_time:
            break
        
        try:
            response = send_request_workspace(record)
            status = "✓" if hasattr(response, 'status_code') and response.status_code < 400 else "✓"
            pbar.set_postfix({"Last": f"{status} Index {index}"})
        except Exception as e:
            pbar.set_postfix({"Last": f"✗ Error at {index}"})
        
        pbar.update(1)

# COMMAND ----------

endpoint_config = workspace.serving_endpoints.get(name="tennis-from-bundle-serving-fe-dev")
print("Endpoint config:", endpoint_config)

# COMMAND ----------

# MAGIC %md
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ## Refresh Monitoring

# COMMAND ----------

from databricks.connect import DatabricksSession
from databricks.sdk import WorkspaceClient

from house_price.config import ProjectConfig
from house_price.monitoring import create_or_refresh_monitoring

spark = DatabricksSession.builder.getOrCreate()
workspace = WorkspaceClient()

# Load configuration
config = ProjectConfig.from_yaml(config_path="../project_config.yml", env="dev")

create_or_refresh_monitoring(config=config, spark=spark, workspace=workspace)
