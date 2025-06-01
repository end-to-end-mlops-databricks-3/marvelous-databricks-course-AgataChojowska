"""Unit tests for DataProcessor."""

import pandas as pd
import pytest
from conftest import CATALOG_DIR

from tennis.config import ProjectConfig
from tennis.data_processor import DataProcessor, split_data


def test_data_ingestion(sample_data: pd.DataFrame) -> None:
    """Test the data ingestion process by checking the shape of the sample data.

    Asserts that the sample data has at least one row and one column.

    :param sample_data: The sample data to be tested
    """
    assert sample_data.shape[0] > 0
    assert sample_data.shape[1] > 0


def test_dataprocessor_init(
    sample_data: pd.DataFrame,
    config: ProjectConfig,
) -> None:
    """Test the initialization of DataProcessor.

    :param sample_data: Sample DataFrame for testing
    :param config: Configuration object for the project
    """
    processor = DataProcessor(raw_data=sample_data, config=config)
    assert isinstance(processor.raw_data, pd.DataFrame)
    assert processor.raw_data.equals(sample_data)

    assert isinstance(processor.config, ProjectConfig)


def test_column_transformations(sample_data: pd.DataFrame, config: ProjectConfig) -> None:
    """Test column transformations performed by the DataProcessor.

    This function checks if specific column transformations are applied correctly.

    :param sample_data: Input DataFrame containing sample data
    :param config: Configuration object for the project
    """
    print(sample_data.head())
    processor = DataProcessor(raw_data=sample_data, config=config)
    processed_data = processor.process_data()

    assert "winner" not in processed_data.columns
    assert "loser" not in processed_data.columns
    assert processed_data["surface"].dtype == "object"


def test_missing_value_handling(sample_data: pd.DataFrame, config: ProjectConfig) -> None:
    """Test missing value handling in the DataProcessor.

    This function verifies that missing values are handled correctly for
    surface and draw_size columns.

    :param sample_data: Input DataFrame containing sample data
    :param config: Configuration object for the project
    """
    processor = DataProcessor(raw_data=sample_data, config=config)
    processed_data = processor.process_data()

    assert processed_data["surface"].isna().sum() == 0
    assert processed_data["draw_size"].isna().sum() == 0


def test_column_selection(sample_data: pd.DataFrame, config: ProjectConfig) -> None:
    """Test column selection in the DataProcessor.

    This function checks if the RESULT column is selected and present in the
    processed DataFrame based on the configuration.

    :param sample_data: Input DataFrame containing sample data
    :param config: Configuration object for the project
    """
    processor = DataProcessor(raw_data=sample_data, config=config)
    processed_data = processor.process_data()
    assert "RESULT" in processed_data.columns


def test_split_data_default_params(stats_data: pd.DataFrame, config: ProjectConfig) -> None:
    """Test the default parameters of the split_data method in DataProcessor.

    This function tests if the split_data method correctly splits the input DataFrame
    into train and test sets using default parameters.

    :param sample_data: Input DataFrame to be split
    :param config: Configuration object for the project
    """
    X_train, X_test, y_train, y_test = split_data(stats_data, config=config)

    assert isinstance(X_train, pd.DataFrame)
    assert isinstance(X_test, pd.DataFrame)
    assert len(X_train) + len(X_test) == len(stats_data)
    assert set(X_train.columns) == set(X_test.columns)

    # # The following lines are just to mimick the behavior of delta tables in UC
    # # Just one time execution in order for all other tests to work
    X_train.to_csv((CATALOG_DIR / "X_train.csv").as_posix(), index=False)  # noqa
    X_test.to_csv((CATALOG_DIR / "X_test.csv").as_posix(), index=False)  # noqa
    y_train.to_csv((CATALOG_DIR / "y_train.csv").as_posix(), index=False)  # noqa
    y_test.to_csv((CATALOG_DIR / "y_test.csv").as_posix(), index=False)  # noqa


def test_preprocess_empty_dataframe(config: ProjectConfig) -> None:
    """Test the preprocess method with an empty DataFrame.

    This function tests if the preprocess method correctly handles an empty DataFrame
    and raises KeyError.

    :param config: Configuration object for the project
    :raises KeyError: If the preprocess method handles empty DataFrame correctly
    """
    processor = DataProcessor(raw_data=pd.DataFrame([]), config=config)
    with pytest.raises(KeyError):
        processor.process_data()
