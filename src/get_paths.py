"""Script for geting system paths."""

import os
from pathlib import Path

from utils.infrastructure import load_csv_to_dataframe, read_yaml_from_path

# get absolute path to project directory from current directory:
PROJECT_DIR = Path(__file__).parents[1]

# get absolute path to parameters directory:
PARAMETERS_DIR = os.path.join(PROJECT_DIR, "parameters")

# get absolute path to data directory:
DATA_DIR = read_yaml_from_path(os.path.join(PROJECT_DIR, "paths", "paths.yml"))[
    "DATA_DIR"
]
