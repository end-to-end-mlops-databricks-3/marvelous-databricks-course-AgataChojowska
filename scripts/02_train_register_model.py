"""Tennis match model training and registration."""

import argparse
import os
from typing import Literal

import mlflow
import pandas as pd
from dotenv import load_dotenv
from loguru import logger
from pyspark.dbutils import DBUtils
from pyspark.sql import SparkSession

from tennis.catalog_utils import load_from_table_to_pandas
from tennis.config import ProjectConfig, Tags
from tennis.models.custom_model import TennisModel
from tennis.runtime_utils import get_spark, running_on_databricks
from tennis.stats_calculator import StatsCalculator

parser = argparse.ArgumentParser()
parser.add_argument(
    "--root_path",
    action="store",
    default=None,
    type=str,
    required=True,
)

parser.add_argument(
    "--env",
    action="store",
    default=None,
    type=str,
    required=True,
)

parser.add_argument(
    "--git_sha",
    action="store",
    default=None,
    type=str,
    required=True,
)

parser.add_argument(
    "--job_run_id",
    action="store",
    default=None,
    type=str,
    required=True,
)

parser.add_argument(
    "--branch",
    action="store",
    default=None,
    type=str,
    required=True,
)


def train_register(custom_model: TennisModel) -> None:
    """Train, log and register custom model."""
    # Prepare features
    custom_model.prepare_features()
    logger.info("Loaded data, prepared features.")

    # Train + log the model (runs everything including MLflow logging)
    custom_model.train()
    custom_model.log_model()
    logger.info("Model training completed.")

    custom_model.register_model()
    logger.info("Registered model")


def predict(config: ProjectConfig, spark: SparkSession, custom_model: TennisModel) -> tuple[float, float]:
    """Get inference data and return predictions."""
    # Get inference match data

    player1 = {
        "Name": "Jannik Sinner",  # Name is not needed, but I wrote it for clarity
        "ID": 206173,  # You can search for the ID in "./data/atp_players.csv"
        "ATP_POINTS": 11000,  # You can find this in the ATP website
        "ATP_RANK": 1,  # You can find this in the ATP website
        "AGE": 23.6,  # You don't need to calculate the age to a point decimal (but the more info the better)
        "HEIGHT": 191,  # This can also be found in "./data/atp_players.csv"
    }

    player2 = {
        "Name": "Carlos Alcaraz",
        "ID": 207989,
        "ATP_POINTS": 5000,
        "ATP_RANK": 3,
        "AGE": 21.6,
        "HEIGHT": 183,
    }

    match = {
        "BEST_OF": 3,  # Set this to 5 if grand slam, otherwise 3 normally
        "DRAW_SIZE": 128,  # Depending on the tournament
        "SURFACE": "Clay",  # Surface of the match. Options are ("Hard", "Clay", "Grass", "Carpet")
    }
    stats = StatsCalculator()
    clean_data = load_from_table_to_pandas(spark=spark, config=config, table="clean_data").drop(
        "update_timestamp_utc", axis=1
    )
    prev_stats = stats.get_updated_stats(clean_data=clean_data)
    output = stats.getStats(player1, player2, match, prev_stats)
    match_data = pd.DataFrame([dict(sorted(output.items()))])
    match_data[["ATP_POINTS_DIFF", "ATP_RANK_DIFF", "HEIGHT_DIFF"]] = match_data[
        ["ATP_POINTS_DIFF", "ATP_RANK_DIFF", "HEIGHT_DIFF"]
    ].astype(float)

    # Get predictions
    prob_player1_wins, prob_player2_wins = custom_model.load_latest_model_and_predict(inference_data=match_data)
    logger.info(f"Probability of {player1['Name']} winning: {prob_player1_wins:.2%}")
    logger.info(f"Probability of {player2['Name']} winning: {prob_player2_wins:.2%}")
    return prob_player1_wins, prob_player2_wins


def main(mode: Literal["train", "predict", "full"]) -> None:
    """Run model training and registering and/or prediction."""
    # 0 Configure tracking uri
    if not running_on_databricks():
        load_dotenv()
        profile = os.environ["PROFILE"]
        mlflow.set_tracking_uri(f"databricks://{profile}")
        mlflow.set_registry_uri(f"databricks-uc://{profile}")

    # 1 Parse args, set config, get tags.
    args = parser.parse_args()
    root_path = args.root_path
    config_path = f"{root_path}/project_config.yml"
    config = ProjectConfig.from_yaml(config_path=config_path, env=args.env)
    tags_dict = {"git_sha": args.git_sha, "branch": args.branch, "job_run_id": args.job_run_id}
    tags = Tags(**tags_dict)

    # 2 Load the data
    spark = get_spark()
    dbutils = DBUtils(spark)
    X_train = load_from_table_to_pandas(spark=spark, config=config, table="train_set").drop(
        "update_timestamp_utc", axis=1
    )
    y_train = load_from_table_to_pandas(spark=spark, config=config, table="train_target").drop(
        "update_timestamp_utc", axis=1
    )
    X_test = load_from_table_to_pandas(spark=spark, config=config, table="test_set").drop(
        "update_timestamp_utc", axis=1
    )
    y_test = load_from_table_to_pandas(spark=spark, config=config, table="test_target").drop(
        "update_timestamp_utc", axis=1
    )
    logger.info("Loaded tables")

    # 3 Initialize model
    custom_model = TennisModel(
        config=config,
        tags=tags,
        spark=spark,
        code_paths=["dist/tennis-0.0.1-py3-none-any.whl"],
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
    )
    logger.info("Model initialized.")

    # 4 Train and/or get predictions
    if mode == "train":
        train_register(custom_model=custom_model)

    if mode == "predict":
        predict(config=config, spark=spark, custom_model=custom_model)

    if mode == "full":
        train_register(custom_model=custom_model)
        predict(config=config, spark=spark, custom_model=custom_model)


if __name__ == "__main__":
    main(mode="full")
