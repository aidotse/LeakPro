import os
import sys
# Load the data, pandas dataframe
import pandas as pd
import numpy as np
import pickle

# Path to the dataset zip file
data_folder = "./data"

project_root = os.path.abspath(os.path.join(os.getcwd(), "../../.."))
sys.path.append(project_root)

from examples.minv.mimic.utils.preprocess_mimic_data import preprocess_data, extract_features_and_split

data_path = os.path.join(data_folder, "processed_data.pkl")

# Load the data
with open(data_path, "rb") as f:
    df = pickle.load(f)

# Now call df.info()
df.info(verbose=True)

# Print count of how many columns begin with "curr"
print("Count of columns starting with 'curr':", sum(df.columns.str.startswith("curr")))
# Print count of how many columns begin with "med"
print("Count of columns starting with 'med':", sum(df.columns.str.startswith("med")))
# Print count of how many columns begin with "50"
print("Count of columns starting with '5':", sum(df.columns.str.startswith("5")))