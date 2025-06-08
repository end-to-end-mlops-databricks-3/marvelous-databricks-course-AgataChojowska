"""Configuration file for the project."""

from typing import Any

import yaml
from pydantic import BaseModel


class ColumnsConfig(BaseModel):
    """Column configurations."""

    winner: list[str]
    loser: list[str]
    required: list[str]


class DataProcessingConfig(BaseModel):
    """Data processing configurations."""

    start_year: int = 1992
    end_year: int = 2025


class Tables(BaseModel):
    """Tables configurations."""

    gold: str
    silver: str
    train: str
    test: str


class ProjectConfig(BaseModel):
    """Represent project configuration parameters loaded from YAML."""

    catalog_name: str
    schema_name: str
    file_path: str
    processing: DataProcessingConfig
    columns: ColumnsConfig
    tables: Tables
    experiment_name_custom: str | None
    parameters: dict[str, Any]
    target_name: str
    features: list[str]
    primary_key_cols: list[str]
    experiment_name_fe: str

    @classmethod
    def from_yaml(cls, config_path: str, env: str = "dev") -> "ProjectConfig":
        """Load and parse configuration settings from a YAML file.

        :param config_path: Path to the YAML configuration file
        :param env: Environment name to load environment-specific settings
        :return: ProjectConfig instance initialized with parsed configuration
        """
        if env not in ["prd", "acc", "dev"]:
            raise ValueError(f"Invalid environment: {env}. Expected 'prd', 'acc', or 'dev'")

        with open(config_path) as f:
            config_dict = yaml.safe_load(f)

            # Extract environment-specific settings
            config_dict["catalog_name"] = config_dict[env]["catalog_name"]
            config_dict["schema_name"] = config_dict[env]["schema_name"]
            config_dict["file_path"] = config_dict[env]["file_path"]

            # Remove environment sections from dict before creating config
            for env_key in ["prd", "acc", "dev"]:
                config_dict.pop(env_key, None)

            return cls(**config_dict)


class Tags(BaseModel):
    """Represents a set of tags for a Git commit.

    Contains information about the Git SHA, branch, and job run ID.
    """

    git_sha: str
    branch: str
    job_run_id: str
