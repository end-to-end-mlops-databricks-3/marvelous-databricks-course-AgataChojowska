# Databricks notebook source
# install dependencies
%pip install -e ..
%pip install git+https://github.com/end-to-end-mlops-databricks-3/marvelous@0.1.0

# COMMAND ----------

#restart python
%restart_python

# COMMAND ----------

# system path update, must be after %restart_python
# caution! This is not a great approach
from pathlib import Path
import sys
sys.path.append(str(Path.cwd().parent / 'src'))

# COMMAND ----------

# A better approach (this file must be present in a notebook folder, achieved via synchronization)
%pip install house_price-1.0.1-py3-none-any.whl

# COMMAND ----------

from pyspark.sql import SparkSession
import mlflow
import dotenv

from tennisprediction.config import ProjectConfig
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from lightgbm import LGBMRegressor
from mlflow.models import infer_signature
from marvelous.common import is_databricks
from dotenv import load_dotenv
import os
from mlflow import MlflowClient
import pandas as pd
from tennisprediction import __version__
from mlflow.utils.environment import _mlflow_conda_env
from databricks import feature_engineering
from databricks.feature_engineering import FeatureFunction, FeatureLookup
from pyspark.errors import AnalysisException
import numpy as np
from datetime import datetime
import boto3


# COMMAND ----------

if not is_databricks():
    load_dotenv()
    profile = os.environ["PROFILE"]
    mlflow.set_tracking_uri(f"databricks://{profile}")
    mlflow.set_registry_uri(f"databricks-uc://{profile}")


config = ProjectConfig.from_yaml(config_path="../project_config.yml", env="dev")

# COMMAND ----------

spark = SparkSession.builder.getOrCreate()
fe = feature_engineering.FeatureEngineeringClient()

train_set = spark.table(f"{config.catalog_name}.{config.schema_name}.train_set")
test_set = spark.table(f"{config.catalog_name}.{config.schema_name}.test_set")

# COMMAND ----------

# create feature table with information about houses

feature_table_name = f"{config.catalog_name}.{config.schema_name}.tennis_features_demo"
lookup_features = ["AGE_DIFF", "DRAW_SIZE", "ATP_POINTS_DIFF"]


# COMMAND ----------

# Option 1: feature engineering client <- This should be completed during preprocessing, before splitting to test & train.
feature_table = fe.create_table(
   name=feature_table_name,
   primary_keys=["Id"],
   df=train_set[["Id"]+lookup_features],
   description="Tennis features table",
)

spark.sql(f"ALTER TABLE {feature_table_name} SET TBLPROPERTIES (delta.enableChangeDataFeed = true)")

fe.write_table(
   name=feature_table_name,
   df=test_set[["Id"]+lookup_features],
   mode="merge",
)

# COMMAND ----------

# create feature table with information about houses
# Option 2: SQL

feature_table_name_sql = f"{config.catalog_name}.{config.schema_name}.tennis_features_demo_sql"

spark.sql(f"""
          CREATE OR REPLACE TABLE {feature_table_name_sql}
          ( Id STRING NOT NULL,
            AGE_DIFF INT NOT NULL, 
            DRAW_SIZE INT NOT NULL, 
            ATP_POINTS_DIFF INT NOT NULL);
          """)
# primary key on Databricks is not enforced!
try:
    spark.sql(f"ALTER TABLE {feature_table_name_sql} ADD CONSTRAINT tennis_pk_demo PRIMARY KEY(Id);")
except AnalysisException:
    pass
spark.sql(f"ALTER TABLE {feature_table_name_sql} SET TBLPROPERTIES (delta.enableChangeDataFeed = true);")
spark.sql(f"""
          INSERT INTO {feature_table_name_sql}
          SELECT Id, AGE_DIFF, DRAW_SIZE, ATP_POINTS_DIFF
          FROM {config.catalog_name}.{config.schema_name}.train_set
          """)
spark.sql(f"""
          INSERT INTO {feature_table_name_sql}
          SELECT Id, AGE_DIFF, DRAW_SIZE, ATP_POINTS_DIFF
          FROM {config.catalog_name}.{config.schema_name}.test_set
          """)

# COMMAND ----------

# create feature function
# docs: https://docs.databricks.com/aws/en/sql/language-manual/sql-ref-syntax-ddl-create-sql-function

# problems with feature functions:
# functions are not versioned
# functions may behave differently depending on the runtime (and version of packages and python)
# there is no way to enforce python version & package versions for the function
# this is only supported from runtime 17
# advised to use only for simple calculations

function_name = f"{config.catalog_name}.{config.schema_name}.calculate_age_diff_in_months"

# COMMAND ----------


# Option 1: with Python
spark.sql(f"""
        CREATE OR REPLACE FUNCTION {function_name}(age_diff DOUBLE)
        RETURNS INT
        LANGUAGE PYTHON AS
        $$
        return age_diff * 12
        $$
        """)

# COMMAND ----------

# it is possible to define simple functions in sql only without python
# Option 2
spark.sql(f"""
        CREATE OR REPLACE FUNCTION {function_name}_sql (age_diff DOUBLE)
        RETURNS INT
        RETURN age_diff * 12;
        """)

# COMMAND ----------

# execute function
spark.sql(f"SELECT {function_name}_sql(1) as age_diff_in_months;")

# COMMAND ----------

# create a training set 
# Basically there is some main dataset and I'm adding more features to it. Here it's dummy version where I just drop the later added columns from the main dataset.
training_set = fe.create_training_set(
    df=train_set.drop("AGE_DIFF", "DRAW_SIZE", "ATP_POINTS_DIFF"),
    label=config.target_name,
    feature_lookups=[
        FeatureLookup(
            table_name=feature_table_name,
            feature_names=["AGE_DIFF", "DRAW_SIZE", "ATP_POINTS_DIFF"],
            lookup_key=["Id"],
                ),
        FeatureFunction(
            udf_name=function_name,
            output_name="age_diff_in_months",
            input_bindings={"age_diff": "AGE_DIFF"},
            ),
    ],
    exclude_columns=["update_timestamp_utc"],
    )

# COMMAND ----------

# Train & register a model
training_df = training_set.load_df().toPandas()
training_df.display()

# COMMAND ----------

X_train = training_df[config.features + ["age_diff_in_months"]]
y_train = training_df[config.target_name]

# COMMAND ----------

from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

pipeline = Pipeline(
            steps=[("scaler", StandardScaler()), ("classifier", XGBClassifier(**config.parameters))]
        )  

pipeline.fit(X_train, y_train)

# COMMAND ----------

# Logging model with Feature Engineering client so that the model can access the features from the features table.
# TODO Question: Can I modify my predict() method here like with Wrapper in custom model?
mlflow.set_experiment("/Shared/tennis-model-fe")
with mlflow.start_run(run_name="tennis-run-model-fe",
                      tags={"git_sha": "1234567890abcd",
                            "branch": "week3"},
                            description="demo run for FE model logging") as run:
    # Log parameters and metrics
    run_id = run.info.run_id
    mlflow.log_param("model_type", "XGBoost with scaling")
    mlflow.log_params(config.parameters)

    # Log the model
    signature = infer_signature(model_input=X_train, model_output=pipeline.predict(X_train))
    fe.log_model(
                model=pipeline,
                flavor=mlflow.sklearn,
                artifact_path="xgboost-pipeline-model-fe",
                training_set=training_set,
                signature=signature,
            )


# COMMAND ----------

model_name = f"{config.catalog_name}.{config.schema_name}.model_fe_demo"
model_version = mlflow.register_model(
    model_uri=f'runs:/{run_id}/xgboost-pipeline-model-fe',
    name=model_name,
    tags={"git_sha": "1234567890abcd"})

# COMMAND ----------

# make predictions
# Does this model add features from feature table to the test data by Id?

features = [f for f in ["Id"] + config.features if f not in lookup_features]

predictions = fe.score_batch(
    model_uri=f"models:/{model_name}/{model_version.version}",
    df=test_set[features]
)

# COMMAND ----------

predictions.select("prediction").show(5)

# COMMAND ----------

# Does this mean that if I modify the primary key in test set, will they no longer be present in the feature table?
# Yes.
# When creating a training set you took: train data + features from feature table looked after using primary keys.
# So the keys need to be the same in train data and features table. 
# If keys change, the features can't be found and are set as None.

# -- Actually I think now it can't find the "helper" data for test data. To what extent is that practical in online deployments? For example, if user submits a request, how can it be ID-ed?
from pyspark.sql.functions import col

features = [f for f in ["Id"] + config.features if f not in lookup_features]
test_set_with_new_id = test_set.select(*features).withColumn(
    "Id",
    (col("Id").cast("long") + 1000000).cast("string")
)

predictions = fe.score_batch(
    model_uri=f"models:/{model_name}/{model_version.version}",
    df=test_set_with_new_id
)

# COMMAND ----------

# make predictions for a non-existing entry -> error!
# So it couldn't find the supportive features for test set, as I changed the test set Ids.
predictions.select("prediction").show(5)

# COMMAND ----------

#"AGE_DIFF", "DRAW_SIZE", "ATP_POINTS_DIFF"
no_age_diff_function = f"{config.catalog_name}.{config.schema_name}.replace_age_diff"
spark.sql(f"""
        CREATE OR REPLACE FUNCTION {no_age_diff_function}(age_diff DOUBLE)
        RETURNS DOUBLE
        LANGUAGE PYTHON AS
        $$
        if age_diff is None:
            return 5
        else:
            return age_diff
        $$
        """)

no_draw_size_function = f"{config.catalog_name}.{config.schema_name}.replace_draw_size"
spark.sql(f"""
        CREATE OR REPLACE FUNCTION {no_draw_size_function}(draw_size BIGINT)
        RETURNS BIGINT
        LANGUAGE PYTHON AS
        $$
        if draw_size is None:
            return 8
        else:
            return draw_size
        $$
        """)

no_atp_points_diff_function = f"{config.catalog_name}.{config.schema_name}.replace_atp_points_diff"
spark.sql(f"""
        CREATE OR REPLACE FUNCTION {no_atp_points_diff_function}(atp_points_diff DOUBLE)
        RETURNS DOUBLE
        LANGUAGE PYTHON AS
        $$
        if atp_points_diff is None:
            return 500
        else:
            return atp_points_diff
        $$
        """)

# COMMAND ----------

# what if we want to replace with a default value if entry is not found
# what if we want to look up value in another table? the logics get complex
# problems that arize: functions/ lookups always get executed (if statement is not possible)
# it can get slow...

# step 1: create 3 feature functions

# step 2: redefine create training set

# try again

# create a training set
training_set = fe.create_training_set(
    df=train_set.drop("AGE_DIFF", "DRAW_SIZE", "ATP_POINTS_DIFF"),
    label=config.target_name,
    feature_lookups=[
        FeatureLookup(
            table_name=feature_table_name,
            feature_names=["AGE_DIFF", "DRAW_SIZE", "ATP_POINTS_DIFF"],
            lookup_key="Id",
            rename_outputs={"AGE_DIFF": "lookup_AGE_DIFF",
                            "DRAW_SIZE": "lookup_DRAW_SIZE",
                            "ATP_POINTS_DIFF": "lookup_ATP_POINTS_DIFF"}
                ),
        FeatureFunction(
            udf_name=no_age_diff_function,
            output_name="AGE_DIFF",
            input_bindings={"age_diff": "lookup_AGE_DIFF"},
            ),
        FeatureFunction(
            udf_name=no_draw_size_function,
            output_name="DRAW_SIZE",
            input_bindings={"draw_size": "lookup_DRAW_SIZE"},
        ),
        FeatureFunction(
            udf_name=no_atp_points_diff_function,
            output_name="ATP_POINTS_DIFF",
            input_bindings={"atp_points_diff": "lookup_ATP_POINTS_DIFF"},
        ),
        FeatureFunction(
            udf_name=function_name,
            output_name="age_diff_in_months",
            input_bindings={"age_diff": "AGE_DIFF"},
            ),
    ],
    exclude_columns=["update_timestamp_utc"],
    )

# COMMAND ----------

mlflow.set_experiment("/Shared/demo-model-fe")
with mlflow.start_run(run_name="demo-run-model-fe",
                      tags={"git_sha": "1234567890abcd",
                            "branch": "week3"},
                            description="demo run for FE model logging") as run:
    # Log parameters and metrics
    run_id = run.info.run_id
    mlflow.log_param("model_type", "XGBoost with scaling")
    mlflow.log_params(config.parameters)

    # Log the model
    signature = infer_signature(model_input=X_train, model_output=pipeline.predict(X_train))
    fe.log_model(
                model=pipeline,
                flavor=mlflow.sklearn,
                artifact_path="xgboost-pipeline-model-fe",
                training_set=training_set,
                signature=signature,
            )
model_name = f"{config.catalog_name}.{config.schema_name}.model_fe_demo"
model_version = mlflow.register_model(
    model_uri=f'runs:/{run_id}/xgboost-pipeline-model-fe',
    name=model_name,
    tags={"git_sha": "1234567890abcd"})

# COMMAND ----------

from pyspark.sql.functions import col

features = [f for f in ["Id"] + config.features if f not in lookup_features]
test_set_with_new_id = test_set.select(*features).withColumn(
    "Id",
    (col("Id").cast("long") + 1000000).cast("string")
)

predictions = fe.score_batch(
    model_uri=f"models:/{model_name}/{model_version.version}",
    df=test_set_with_new_id
)

# COMMAND ----------

# make predictions for a non-existing entry -> no error!
predictions.select("prediction").show(5)

# COMMAND ----------

import boto3
import os

region_name = "eu-west-1"
aws_access_key_id = os.environ["aws_access_key_id"]
aws_secret_access_key = os.environ["aws_secret_access_key"]

client = boto3.client(
    'dynamodb',
    aws_access_key_id=aws_access_key_id,
    aws_secret_access_key=aws_secret_access_key,
    region_name=region_name
)

# COMMAND ----------

response = client.create_table(
    TableName='TennisFeatures',
    KeySchema=[
        {
            'AttributeName': 'Id',
            'KeyType': 'HASH'  # Partition key
        }
    ],
    AttributeDefinitions=[
        {
            'AttributeName': 'Id',
            'AttributeType': 'S'  # String
        }
    ],
    ProvisionedThroughput={
        'ReadCapacityUnits': 5,
        'WriteCapacityUnits': 5
    }
)

print("Table creation initiated:", response['TableDescription']['TableName'])

# COMMAND ----------

client.put_item(
    TableName='TennisFeatures',
    Item={
        'Id': {'S': '121212'},
        'AGE_DIFF': {'N': '8'},
        'DRAW_SIZE': {'N': '4'},
        'ATP_POINTS_DIFF': {'N': '200'}
    }
)

# COMMAND ----------

response = client.get_item(
    TableName='TennisFeatures',
    Key={
        'Id': {'S': '121212'}
    }
)

# Extract the item from the response
item = response.get('Item')
print(item)

# COMMAND ----------

from itertools import islice

feature_table_name = f"{config.catalog_name}.{config.schema_name}.tennis_features_demo"

rows = spark.table(feature_table_name).toPandas().to_dict(orient="records")

def to_dynamodb_item(row):
    return {
        'PutRequest': {
            'Item': {
                'Id': {'S': str(row['Id'])},
                'AGE_DIFF': {'N': str(row['AGE_DIFF'])},
                'DRAW_SIZE': {'N': str(row['DRAW_SIZE'])},
                'ATP_POINTS_DIFF': {'N': str(row['ATP_POINTS_DIFF'])}
            }
        }
    }

items = [to_dynamodb_item(row) for row in rows]

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

for batch in chunks(items, 25):
    response = client.batch_write_item(
        RequestItems={
            'TennisFeatures': batch
        }
    )
    # Handle any unprocessed items if needed
    unprocessed = response.get('UnprocessedItems', {})
    if unprocessed:
        print("Warning: Some items were not processed. Retry logic needed.")

# COMMAND ----------

# We ran into more limitations when we tried complex data types as output of a feature function
# and then tried to use it for serving
# al alternatve solution: using an external database (we use DynamoDB here)

# create a DynamoDB table
# insert records into dynamo DB & read from dynamoDB

# create a pyfunc model

# COMMAND ----------


class TennisModelWrapper(mlflow.pyfunc.PythonModel):
    """Wrapper class for machine learning models to be used with MLflow.

    This class wraps a machine learning model for predicting tennis matches.
    """

    def __init__(self, model: object) -> None:
        """Initialize the TennisModelWrapper.

        :param model: The underlying machine learning model.
        """
        self.model = model

    def predict(
        self, context: mlflow.pyfunc.PythonModelContext, model_input: pd.DataFrame | np.ndarray
    ) -> dict[str, float]:
        """Make predictions using the wrapped model.

        :param context: The MLflow context (unused in this implementation).
        :param model_input: Input data for making predictions.
        :return: A dictionary containing the adjusted prediction.
        """
        client = boto3.client('dynamodb',
                                   aws_access_key_id=os.environ["aws_access_key_id"],
                                   aws_secret_access_key=os.environ["aws_secret_access_key"],
                                   region_name=os.environ["region_name"])

        parsed = []
        for lookup_id in model_input["Id"]:
            raw_item = client.get_item(
                TableName='TennisFeatures',
                Key={'Id': {'S': lookup_id}})["Item"]
            parsed_dict = {key: float(value['N']) if 'N' in value else value['S']
                      for key, value in raw_item.items()}
            parsed.append(parsed_dict)
        lookup_df=pd.DataFrame(parsed)
        merged_df = model_input.merge(lookup_df, on="Id", how="left").drop("Id", axis=1)

        merged_df["AGE_DIFF"] = merged_df["AGE_DIFF"].fillna(2)
        merged_df["DRAW_SIZE"] = merged_df["DRAW_SIZE"].fillna(8)
        merged_df["ATP_POINTS_DIFF"] = merged_df["ATP_POINTS_DIFF"].fillna(200)
        merged_df["age_diff_in_months"] = merged_df["AGE_DIFF"] * 12
        predictions = self.model.predict(merged_df)

        return [float(x) for x in predictions]

# COMMAND ----------

from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

pipeline = Pipeline(
            steps=[("scaler", StandardScaler()), ("classifier", XGBClassifier(**config.parameters))]
        )  

pipeline.fit(X_train, y_train)

# COMMAND ----------

custom_model = TennisModelWrapper(pipeline)

# COMMAND ----------

features = [f for f in ["Id"] + config.features if f not in lookup_features]
data = test_set.select(*features).toPandas()
data

# COMMAND ----------

X_train

# COMMAND ----------

# Reorder by listing columns in desired order
# data = data[["AGE_DIFF", "ATP_POINTS_DIFF", "ATP_RANK_DIFF", "BEST_OF", "DRAW_SIZE", "ELO_DIFF", "ELO_GRAD_LAST_100_DIFF", "ELO_GRAD_LAST_10_DIFF", "ELO_GRAD_LAST_200_DIFF", "ELO_GRAD_LAST_25_DIFF", "ELO_GRAD_LAST_3_DIFF", "ELO_GRAD_LAST_50_DIFF", "ELO_GRAD_LAST_5_DIFF", "ELO_SURFACE_DIFF", "H2H_DIFF", "H2H_SURFACE_DIFF", "HEIGHT_DIFF", "N_GAMES_DIFF", "P_1ST_IN_LAST_100_DIFF", "P_1ST_IN_LAST_10_DIFF", "P_1ST_IN_LAST_200_DIFF", "P_1ST_IN_LAST_25_DIFF", "P_1ST_IN_LAST_3_DIFF", "P_1ST_IN_LAST_50_DIFF", "P_1ST_IN_LAST_5_DIFF", "P_1ST_WON_LAST_100_DIFF", "P_1ST_WON_LAST_10_DIFF", "P_1ST_WON_LAST_200_DIFF", "P_1ST_WON_LAST_25_DIFF", "P_1ST_WON_LAST_3_DIFF", "P_1ST_WON_LAST_50_DIFF", "P_1ST_WON_LAST_5_DIFF", "P_2ND_WON_LAST_100_DIFF", "P_2ND_WON_LAST_10_DIFF", "P_2ND_WON_LAST_200_DIFF", "P_2ND_WON_LAST_25_DIFF", "P_2ND_WON_LAST_3_DIFF", "P_2ND_WON_LAST_50_DIFF", "P_2ND_WON_LAST_5_DIFF", "P_ACE_LAST_100_DIFF", "P_ACE_LAST_10_DIFF", "P_ACE_LAST_200_DIFF", "P_ACE_LAST_25_DIFF", "P_ACE_LAST_3_DIFF", "P_ACE_LAST_50_DIFF", "P_ACE_LAST_5_DIFF", "P_BP_SAVED_LAST_100_DIFF", "P_BP_SAVED_LAST_10_DIFF", "P_BP_SAVED_LAST_200_DIFF", "P_BP_SAVED_LAST_25_DIFF", "P_BP_SAVED_LAST_3_DIFF", "P_BP_SAVED_LAST_50_DIFF", "P_BP_SAVED_LAST_5_DIFF", "P_DF_LAST_100_DIFF", "P_DF_LAST_10_DIFF", "P_DF_LAST_200_DIFF", "P_DF_LAST_25_DIFF", "P_DF_LAST_3_DIFF", "P_DF_LAST_50_DIFF", "P_DF_LAST_5_DIFF", "WIN_LAST_100_DIFF", "WIN_LAST_10_DIFF", "WIN_LAST_200_DIFF", "WIN_LAST_25_DIFF", "WIN_LAST_3_DIFF", "WIN_LAST_50_DIFF", "WIN_LAST_5_DIFF", "age_diff_in_months"]]
custom_model.predict(context=None, model_input=data)

# COMMAND ----------

#log model
mlflow.set_experiment("/Shared/demo-model-fe-pyfunc")
with mlflow.start_run(run_name="demo-run-model-fe-pyfunc",
                      tags={"git_sha": "1234567890abcd",
                            "branch": "week2"},
                            description="demo run for FE model logging") as run:
    # Log parameters and metrics
    run_id = run.info.run_id
    mlflow.log_param("model_type", "LightGBM with preprocessing")
    mlflow.log_params(config.parameters)

    # Log the model
    signature = infer_signature(model_input=data, model_output=custom_model.predict(context=None, model_input=data))
    mlflow.pyfunc.log_model(
                python_model=custom_model,
                artifact_path="lightgbm-pipeline-model-fe",
                signature=signature,
            )


# COMMAND ----------

# predict
mlflow.models.predict(f"runs:/{run_id}/lightgbm-pipeline-model-fe", data[0:1])
