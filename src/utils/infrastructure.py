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

def save_to_yaml(
        path: str,
        dictionary: dict,
):
    """
    Save given `dictionary` to a yaml file stored in `filepath`, e.g. '/home/per/medsensio/learning/tensorflow/metrics/metrics.yml'.
    """
    dirpath = os.path.dirname(path)
    # make directory in case it does not exist:
    if not os.path.exists(dirpath):
        os.mkdir(dirpath)
    with open(path, 'w', encoding='utf-8') as f:
        yaml.dump(dictionary, f)

