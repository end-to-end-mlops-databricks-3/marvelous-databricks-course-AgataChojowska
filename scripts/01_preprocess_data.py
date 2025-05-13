"""Tennis match data processing pipeline."""

import yaml
from loguru import logger
from marvelous.timer import Timer

from tennis.catalog_utils import save_to_catalog
from tennis.config import ProjectConfig
from tennis.data_processor import DataProcessor
from tennis.runtime_utils import setup_project_logging


def main() -> None:
    """Execute the data processing pipeline."""
    config_path = "project_config.yml"

    # Load configuration
    config = ProjectConfig.from_yaml(config_path=config_path, env="dev")

    # Set up logging with configuration
    setup_project_logging(config)

    logger.info("Configuration loaded:")
    logger.info(yaml.dump(config, default_flow_style=False))

    # Preprocess the data
    with Timer() as preprocess_timer:
        data_processor = DataProcessor(config)
        processed_data = data_processor.process_data()

        logger.info(f"Processed Data Shape: {processed_data.shape}")
        logger.info(f"Columns: {processed_data.columns.tolist()}")
        logger.info("First few rows:")
        logger.info(f"\n{processed_data.head()}")

    logger.info(f"Data preprocessing completed in: {preprocess_timer}")

    # Split the data
    X_train, X_test = data_processor.split_data(processed_data)
    logger.info("Training set shape: %s", X_train.shape)
    logger.info("Test set shape: %s", X_test.shape)

    # Save to catalog
    logger.info("Saving data to catalog")
    datasets = {"train_set": X_train, "test_set": X_test}
    for table_name, dataset in datasets.items():
        save_to_catalog(dataset=dataset, config=config, table_name=table_name)


if __name__ == "__main__":
    main()
