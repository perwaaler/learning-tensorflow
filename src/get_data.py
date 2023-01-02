"""
A script for getting the data that contains insurance data; generates a
pandas dataframe object.
"""

import os

import pandas as pd

from utils.infrastructure import load_csv_to_dataframe
from get_paths import DATA_DIR

DATA = load_csv_to_dataframe(os.path.join(DATA_DIR, "insurance.csv"))