"""Tennis Data Processor for match outcome prediction."""

import numpy as np
import pandas as pd
from loguru import logger
from pyspark.sql import SparkSession
from sklearn.model_selection import train_test_split

from tennis.config import ProjectConfig


class DataProcessor:
    """Process tennis match data for machine learning.

    Handles data loading, cleaning, and feature engineering for tennis match prediction.
    Supports both local and Databricks runtime environments.

    """

    def __init__(self, config: ProjectConfig, spark: SparkSession) -> None:
        self.config = config
        self.spark = spark

    def load_data(self) -> pd.DataFrame:
        """Load match data from CSV files in Unity Catalog.

        Args:
            start_year: First year of data to load. Defaults to 1992.
            end_year: Last year of data to load (exclusive). Defaults to 2024.

        Returns:
            DataFrame containing all match data from specified years.

        """
        file_pattern = f"/Volumes/{self.config.catalog_name}/{self.config.schema_name}/{self.config.file_path}"

        df_spark = self.spark.read.csv(file_pattern, header=True, inferSchema=True)

        df_pandas = df_spark.toPandas()

        df_pandas["year"] = df_pandas["tourney_date"].astype(str).str[:4].astype(int)
        df_filtered = df_pandas[
            (df_pandas["year"] >= self.config.processing.start_year)
            & (df_pandas["year"] < self.config.processing.end_year)
        ]

        return df_filtered

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove rows with missing values in critical columns."""
        df_cleaned = df.dropna(subset=self.config.columns.required)
        df_cleaned = df_cleaned.reset_index(drop=True)
        return df_cleaned

    def randomize_players(self, df: pd.DataFrame) -> pd.DataFrame:
        """Randomize winner/loser assignments to player1/player2 positions.

        Creates RESULT column: 1 if player1 won, 0 if player2 won.
        """
        df_randomized = df.copy()

        rename_dict = {col: col.replace("winner", "p1").replace("w_", "p1_") for col in self.config.columns.winner}
        rename_dict.update({col: col.replace("loser", "p2").replace("l_", "p2_") for col in self.config.columns.loser})

        df_randomized = df_randomized.rename(columns=rename_dict)

        # Generate boolean mask for 50% of rows (True means swap)
        mask = np.random.rand(len(df_randomized)) < 0.5

        # Identify player columns
        player1_cols = [col for col in df_randomized.columns if "player1" in col or "p1_" in col]
        player2_cols = [col for col in df_randomized.columns if "player2" in col or "p2_" in col]

        # Create RESULT column (1 = not swapped, 0 = swapped)
        df_randomized["RESULT"] = np.where(mask, 0, 1)

        # Swap values where mask is True
        df_randomized.loc[mask, player1_cols], df_randomized.loc[mask, player2_cols] = (
            df_randomized.loc[mask, player2_cols].values,
            df_randomized.loc[mask, player1_cols].values,
        )

        return df_randomized

    def process_data(self) -> pd.DataFrame:
        """Execute full data processing pipeline.

        Args:
            start_year: First year of data to process. Defaults to 1992.
            end_year: Last year of data to process (exclusive). Defaults to 2024.

        """
        logger.info(
            f"Loading ATP match data from {self.config.processing.start_year} to {self.config.processing.end_year}..."
        )
        df = self.load_data()
        logger.info(f"Loaded {len(df)} matches")

        logger.info("Cleaning data...")
        df_cleaned = self.clean_data(df)
        logger.info(f"Cleaned data: {len(df_cleaned)} matches remaining")

        logger.info("Randomizing player assignments...")
        df_processed = self.randomize_players(df_cleaned)
        logger.info(f"Final dataset: {len(df_processed)} matches")

        return df_processed

    def split_data(
        self, df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Split the DataFrame (self.df) into training and test sets.

        :param test_size: The proportion of the dataset to include in the test split.
        :param random_state: Controls the shuffling applied to the data before applying the split.
        :return: A tuple containing the training and test DataFrames.
        """
        train_set, test_set = train_test_split(df, test_size=test_size, random_state=random_state)
        return train_set, test_set
