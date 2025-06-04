"""Functions that allow for seamless running of code and logging both on Databricks as well as locally."""

import os
from typing import Union

from databricks.connect import DatabricksSession
from loguru import logger
from marvelous.logging import setup_logging
from pyspark.sql import SparkSession

from tennis.config import ProjectConfig


def running_on_databricks() -> bool:
    """Check if code is running in Databricks runtime environment."""
    return "DATABRICKS_RUNTIME_VERSION" in os.environ


def get_spark(profile: str = "dev") -> Union[SparkSession, DatabricksSession]:
    """Get appropriate Spark session based on runtime environment."""
    if running_on_databricks():
        return SparkSession.builder.getOrCreate()
    return DatabricksSession.builder.profile(profile).getOrCreate()


def setup_project_logging(config: ProjectConfig) -> None:
    """Configure logging based on runtime environment."""
    logger.remove()

    if running_on_databricks():
        setup_logging(log_file=f"/Volumes/{config.catalog_name}/{config.schema_name}/logs/marvelous-1.log")
    else:
        local_log_dir = os.path.expanduser("~/logs")
        os.makedirs(local_log_dir, exist_ok=True)
        log_file = f"{local_log_dir}/databricks_connect.log"

        logger.add(log_file, rotation="500 MB", retention="10 days")
