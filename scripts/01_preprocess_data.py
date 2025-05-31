"""Tennis match data processing pipeline."""

import yaml
from loguru import logger
from marvelous.timer import Timer

from tennis.catalog_utils import load_csv, save_to_catalog
from tennis.config import ProjectConfig
from tennis.data_processor import DataProcessor
from tennis.runtime_utils import get_spark, setup_project_logger
from tennis.stats_calculator import StatsCalculator


def main() -> None:
    """Execute the data processing pipeline."""
    config_path = "project_config.yml"

    # Load configuration
    config = ProjectConfig.from_yaml(config_path=config_path, env="dev")

    # Set up logger with configuration
    setup_project_logger(config)

    logger.info("Configuration loaded:")
    logger.info(yaml.dump(config, default_flow_style=False))

    # Get SparkSession or Databricks session based on your current runtime.
    spark = get_spark()

    # Preprocess the data
    with Timer() as preprocess_timer:
        raw_data = load_csv(config=config, spark=spark)
        data_processor = DataProcessor(raw_data=raw_data, config=config)
        clean_data = data_processor.process_data()

        logger.info(f"Processed Data Shape: {clean_data.shape}")
        logger.info("First few rows:")
        logger.info(f"\n{clean_data.head()}")
        logger.info("Columns")
        logger.info(clean_data.columns)

        stats = StatsCalculator()
        stats_data = stats.get_dataset_with_stats(clean_data=clean_data)

        logger.info(f"Processed Stats Data Shape: {stats_data.shape}")
        logger.info("First few rows:")
        logger.info(f"\n{stats_data.head()}")

    logger.info(f"Data preprocessing completed in: {preprocess_timer}")

    # Split the data
    X_train, X_test, y_train, y_test = data_processor.split_data(df=stats_data, target_name="RESULT")
    logger.info("Training set shape: %s", X_train.shape)
    logger.info("Test set shape: %s", X_test.shape)
    logger.info("Target train shape: %s", y_train.shape)
    logger.info("Target test shape: %s", y_test.shape)

    # Save to catalog
    logger.info("Saving cleaned data to catalog")
    save_to_catalog(dataset=clean_data, config=config, spark=spark, table_name="clean_data")

    logger.info("Saving stats data to catalog")
    datasets = {"train_set": X_train, "test_set": X_test, "train_target": y_train, "test_target": y_test}
    for table_name, dataset in datasets.items():
        save_to_catalog(dataset=dataset, config=config, spark=spark, table_name=table_name)


if __name__ == "__main__":
    main()
