import os
import pickle
import sys

# Load the data, pandas dataframe

# Path to the dataset zip file
data_folder = "./data"

project_root = os.path.abspath(os.path.join(os.getcwd(), "../../.."))
sys.path.append(project_root)


data_path = os.path.join(data_folder, "processed_data.pkl")

# Load the data
with open(data_path, "rb") as f:
    df = pickle.load(f)

# Now call df.info()
df.info(verbose=True)

# Print count of how many columns begin with "curr"
# Print count of how many columns begin with "med"
# Print count of how many columns begin with "50"
