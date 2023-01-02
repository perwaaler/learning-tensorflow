"""
Module that contains functions relating to loading, setting paths, etc...
"""
import os

from pathlib import Path

import yaml
import pandas as pd


def read_yaml_from_path(
    path: str,
) -> dict:
    """Read yaml file from `path`."""

    with open(path, mode="r", encoding="utf-8") as jfl:
        yaml_loaded = yaml.safe_load(jfl)

    return yaml_loaded


def load_csv_to_dataframe(data_path):
    """Loads a data"""
    df = pd.read_csv(data_path)
    return df
