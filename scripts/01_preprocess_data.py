"""Tennis match data processing pipeline."""

import yaml
from loguru import logger
from marvelous.common import create_parser
from marvelous.timer import Timer

from tennisprediction.catalog_utils import load_csv, save_to_catalog
from tennisprediction.config import ProjectConfig
from tennisprediction.data_processor import DataProcessor, split_data
from tennisprediction.runtime_utils import get_spark
from tennisprediction.stats_calculator import StatsCalculator


def main() -> None:
    """Execute the data processing pipeline."""
    args = create_parser()
    config_path = "project_config.yml"

    # Load configuration
    root_path = args.root_path
    config_path = f"{root_path}/files/project_config.yml"
    config = ProjectConfig.from_yaml(config_path=config_path, env=args.env)
    is_test = args.is_test

    logger.info("Configuration loaded:")
    logger.info(yaml.dump(config, default_flow_style=False))

    # Get SparkSession or Databricks session based on your current runtime.
    spark = get_spark()

    # Preprocess the data
    with Timer() as preprocess_timer:
        raw_data = load_csv(config=config, spark=spark)
        data_processor = DataProcessor(raw_data=raw_data, config=config)
        clean_data = data_processor.process_data()

        print(f"Processed Data Shape: {clean_data.shape}")
        print("First few rows:")
        print(f"\n{clean_data.head()}")
        print("Columns")
        print(clean_data.columns)

        stats = StatsCalculator()
        stats_data = stats.get_dataset_with_stats(clean_data=clean_data)

        print(f"Processed Stats Data Shape: {stats_data.shape}")
        print("First few rows:")
        print(f"\n{stats_data.head()}")
        stats_data.to_csv("stats_data.csv")

    print(f"Data preprocessing completed in: {preprocess_timer}")

    # Split the data
    X_train, X_test = split_data(df=stats_data)
    print("Training set shape: %s", X_train.shape)
    print("Test set shape: %s", X_test.shape)

    # Save to catalog
    print("Saving cleaned data to catalog")
    save_to_catalog(dataset=clean_data, config=config, spark=spark, table_name=config.tables.silver)

    print("Saving stats data to catalog")
    save_to_catalog(dataset=stats_data, config=config, spark=spark, table_name=config.tables.gold)

    print("Saving train and test data to catalog")
    datasets = {"train_set": X_train, "test_set": X_test}
    for table_name, dataset in datasets.items():
        save_to_catalog(dataset=dataset, config=config, spark=spark, table_name=table_name)


if __name__ == "__main__":
    main()
