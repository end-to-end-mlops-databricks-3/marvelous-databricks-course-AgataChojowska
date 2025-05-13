"""Unity Catalog utilities for saving datasets to Databricks."""

import pandas as pd
from loguru import logger
from pyspark.sql.functions import current_timestamp, to_utc_timestamp

from tennis.config import ProjectConfig
from tennis.runtime_utils import get_spark


def save_to_catalog(dataset: pd.DataFrame, config: ProjectConfig, table_name: str, optimize: bool = True) -> None:
    """Save dataset as Delta tables in Databricks catalog with timestamps."""
    logger.info("Saving train and test sets to catalog as Delta tables")

    try:
        spark = get_spark()
        dataset_with_timestamp = spark.createDataFrame(dataset).withColumn(
            "update_timestamp_utc", to_utc_timestamp(current_timestamp(), "UTC")
        )

        # Save as Delta tables with explicit format
        table_name = f"{config.catalog_name}.{config.schema_name}.{table_name}"

        dataset_with_timestamp.write.mode("overwrite").format("delta").option("overwriteSchema", "true").saveAsTable(
            table_name
        )

        logger.success(f"Successfully saved Delta tables: {table_name}")

        if optimize:
            spark.sql(f"OPTIMIZE {table_name}")

    except Exception as e:
        logger.error(f"Error saving to catalog: {str(e)}")
        raise
