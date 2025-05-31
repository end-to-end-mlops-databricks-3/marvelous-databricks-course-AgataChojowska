"""Unity Catalog utilities for handling data in Databricks."""

import pandas as pd
from loguru import logger
from pyspark.sql import SparkSession
from pyspark.sql.functions import current_timestamp, to_utc_timestamp

from tennis.config import ProjectConfig


def save_to_catalog(
    dataset: pd.DataFrame, config: ProjectConfig, spark: SparkSession, table_name: str, optimize: bool = True
) -> None:
    """Save dataset as Delta tables in Databricks catalog with timestamps."""
    logger.info("Saving train and test sets to catalog as Delta tables")

    try:
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


def load_csv(config: ProjectConfig, spark: SparkSession, header: bool = True, inferSchema: bool = True) -> pd.DataFrame:
    """Load data from CSV files in Unity Catalog to pandas dataframe."""
    file_pattern = f"/Volumes/{config.catalog_name}/{config.schema_name}/{config.file_path}"
    df_spark = spark.read.csv(file_pattern, header=header, inferSchema=inferSchema)
    return df_spark.toPandas()


def load_from_table_to_pandas(
    spark: SparkSession, config: ProjectConfig, table: str, limit: int | None = None
) -> pd.DataFrame:
    """Load data from Databricks tables to pandas Dataframe."""
    dataset_spark = spark.table(f"{config.catalog_name}.{config.schema_name}.{table}")  # TODO try with limit as query
    if limit:
        dataset_spark = dataset_spark.limit(limit)
    dataset = dataset_spark.toPandas()
    data_version = "0"  # describe history -> retrieve
    logger.info(f"✅ Data from table {table}, schema {config.schema_name} successfully loaded.")
    return dataset


def load_from_table_to_spark(
    spark: SparkSession, config: ProjectConfig, table: str, limit: int | None = None
) -> pd.DataFrame:
    """Load data from Databricks tables to spark Dataframe."""
    dataset_spark = spark.table(f"{config.catalog_name}.{config.schema_name}.{table}")
    if limit:
        dataset_spark = dataset_spark.limit(limit)
    data_version = "0"  # describe history -> retrieve
    logger.info(f"✅ Data from table {table}, schema {config.schema_name} successfully loaded.")
    return dataset_spark


def load_sample_data_to_pandas(
    spark: SparkSession, config: ProjectConfig, table: str, limit: int = 100
) -> pd.DataFrame:
    """Load sample data from Databricks tables to pandas Dataframe."""
    query = f"""
    SELECT * FROM {config.catalog_name}.{config.schema_name}.{table}
    LIMIT {limit}
    """
    dataset_spark = spark.sql(query)
    dataset = dataset_spark.toPandas()
    logger.info(f"✅ Sample data loaded ({len(dataset)} rows from {table}).")
    return dataset
