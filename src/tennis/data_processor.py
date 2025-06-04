"""Tennis Data Processor for match outcome prediction."""

import re

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.model_selection import train_test_split

from tennis.config import ProjectConfig


class DataProcessor:
    """Process tennis match data for machine learning.

    Handles data loading, cleaning, and feature engineering for tennis match prediction.
    Supports both local and Databricks runtime environments.

    """

    def __init__(self, raw_data: pd.DataFrame, config: ProjectConfig) -> None:
        self.raw_data = raw_data
        self.config = config

    def filter_data(self) -> pd.DataFrame:
        """Filter data to create a dataFrame containing all match data from specified years."""
        self.raw_data["year"] = self.raw_data["tourney_date"].astype(str).str[:4].astype(int)
        df_filtered = self.raw_data[
            (self.raw_data["year"] >= self.config.processing.start_year)
            & (self.raw_data["year"] < self.config.processing.end_year)
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

        # Convert columns to object dtype to allow mixed types
        player1_cols_obj = [col for col in player1_cols]
        player2_cols_obj = [col for col in player2_cols]

        for col1, col2 in zip(player1_cols_obj, player2_cols_obj):
            df_randomized[col1] = df_randomized[col1].astype("object")
            df_randomized[col2] = df_randomized[col2].astype("object")

        # Now perform the swap
        df_randomized.loc[mask, player1_cols], df_randomized.loc[mask, player2_cols] = (
            df_randomized.loc[mask, player2_cols].values,
            df_randomized.loc[mask, player1_cols].values,
        )

        return df_randomized

    def replace_letters_with_numbers(self, text: str) -> str:
        """Replace each letter with its position in alphabet, keep numbers same."""
        if pd.isna(text):
            return text

        def letter_to_number(match: str) -> str:
            """Change letter to number with its position in alphabet."""
            letter = match.group(0).upper()
            return str(ord(letter) - ord("A") + 1)

        return re.sub(r"[A-Za-z]", letter_to_number, str(text))

    def text_cols_to_numbers_cols(self, df: pd.DataFrame, col_name: str = "Id") -> pd.DataFrame:
        """Replace rach letter with its position in alphabet in the whole column."""
        df[col_name] = df[col_name].apply(self.replace_letters_with_numbers)
        return df

    def merge_cols_into_one(self, df: pd.DataFrame, new_col_name: str = "Id", sep: str = "") -> pd.DataFrame:
        """Merge contents of 2 columns into one new column."""
        df[new_col_name] = df[self.config.primary_key_cols].astype(str).agg(sep.join, axis=1)
        df[new_col_name] = df[new_col_name].str.replace("-", "")
        return df

    def process_data(self) -> pd.DataFrame:
        """Execute full data processing pipeline."""
        logger.info(
            f"Loading ATP match data from {self.config.processing.start_year} to {self.config.processing.end_year}..."
        )
        df = self.filter_data()
        logger.info(f"Loaded {len(df)} matches")

        logger.info("Cleaning data...")
        df_cleaned = self.clean_data(df)
        logger.info(f"Cleaned data: {len(df_cleaned)} matches remaining")

        logger.info("Randomizing player assignments...")
        df_processed = self.randomize_players(df_cleaned)

        logger.info("Getting the primary key column...")
        df_merged = self.merge_cols_into_one(df_processed)
        df_final = self.text_cols_to_numbers_cols(df_merged)

        logger.info(f"Final dataset: {len(df_final)} matches")
        logger.info(df_final.columns)

        return df_final


def split_data(
    df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split the DataFrame (self.df) into training and test sets.

    :param test_size: The proportion of the dataset to include in the test split.
    :param random_state: Controls the shuffling applied to the data before applying the split.
    :return: A tuple containing the training and test DataFrames.
    """
    train_set, test_set = train_test_split(df, test_size=test_size, random_state=random_state)
    return train_set, test_set
