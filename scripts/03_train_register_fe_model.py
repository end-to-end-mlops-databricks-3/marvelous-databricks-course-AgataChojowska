from loguru import logger
from pyspark.sql import SparkSession
from marvelous.common import create_parser
from pyspark.dbutils import DBUtils

from tennisprediction.config import ProjectConfig, Tags
from tennisprediction.models.feature_lookup_model import FeatureLookUpModel

args = create_parser()

root_path = args.root_path
config_path = f"{root_path}/files/project_config.yml"

config = ProjectConfig.from_yaml(config_path=config_path, env=args.env)
spark = SparkSession.builder.getOrCreate()
dbutils = DBUtils(spark)
tags_dict = {"git_sha": args.git_sha, "branch": args.branch, "job_run_id": args.job_run_id}
tags = Tags(**tags_dict)



# Initialize model
spark = SparkSession.builder.getOrCreate()
fe_model = FeatureLookUpModel(config=config, tags=tags, spark=spark)
logger.info("Model initialized.")


# Create feature table
# fe_model.create_feature_table()

# Update feature table
fe_model.update_feature_table()
logger.info("Feature table updated.")


# Define house age feature function
# fe_model.define_feature_function()


# Load data
fe_model.load_data()
logger.info("Loaded the data")


# Perform feature engineering
fe_model.feature_engineering()
logger.info("Done with Feature Engineering.")

# Train the model
fe_model.train()
logger.info("Model training completed.")

# Model evaluation could be done here

# Register the model
fe_model.register_model()

is_test = args.is_test

# when running test, always register and deploy (or always as I don't have eval)
model_improved = True


if model_improved:
    # Register the model
    latest_version = fe_model.register_model()
    logger.info("New model registered with version:", latest_version)
    dbutils.jobs.taskValues.set(key="model_version", value=latest_version)
    dbutils.jobs.taskValues.set(key="model_updated", value=1)

else:
    dbutils.jobs.taskValues.set(key="model_updated", value=0)
