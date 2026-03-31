import pandas as pd
import numpy as np
import os
import yaml
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
from pathlib import Path

def preprocess_data(input_path, lab_events_path, continuous_col_names, mean_imputation=False):
    """
    Preprocess the MIMIC dataset by handling missing values, removing outliers,
    encoding categorical variables, and preparing the dataset for modeling.

    Args:
        input_path (str): Path to the input pickle file containing the main dataset.
        lab_events_path (str): Path to the pickle file containing lab events data.
        continuous_col_names (list): List of column names that are continuous variables.
        mean_imputation (bool): If True, use mean imputation for missing values in continuous columns.
            If False, use random sampling from the same column's distribution. Valid for columns with normal distributions.

    Returns:
        output_path (str): Path to the processed dataset saved as a pickle file.
        Saves the processed DataFrame to a pickle file in same folder as input_path.
    """

    df = pd.read_pickle(input_path)
    df_lab_events = pd.read_pickle(lab_events_path)

    # Get copy of df, we don't want the original df to be modified (avoids pandas slice warning)
    df_processed = df.copy()

    # Replace null values in "Height (Inches)" with values from "Height". Both are on the same scale
    df_processed["Height (Inches)"] = df_processed["Height (Inches)"].fillna(df_processed["Height"])
    # Remove extreme outliers, for example, there exists 9219.5 inches for height (Inches)
    min_height = 0
    max_height = 100 
    df_processed = df_processed[(df_processed["Height (Inches)"] > min_height) & (df_processed["Height (Inches)"] < max_height) | df_processed["Height (Inches)"].isnull()]

    # Replace null values in "Weight (Lbs)" with values from "Weight". Both are on the same scale
    df_processed["Weight (Lbs)"] = df_processed["Weight (Lbs)"].fillna(df_processed["Weight"])
    # Remove extreme outliers
    min_weight = 0
    max_weight = 700
    df_processed = df_processed[(df_processed["Weight (Lbs)"] > min_weight) & (df_processed["Weight (Lbs)"] < max_weight) | df_processed["Weight (Lbs)"].isnull()]

    # Replace null values in BMI (kg/m2) with values from BMI. Both are on the same scale (not sure why)
    df_processed["BMI (kg/m2)"] = df_processed["BMI (kg/m2)"].fillna(df_processed["BMI"])
    # Remove extreme outliers
    min_bmi = 0
    max_bmi = 100
    df_processed = df_processed[(df_processed["BMI (kg/m2)"] > min_bmi) & (df_processed["BMI (kg/m2)"] < max_bmi) | df_processed["BMI (kg/m2)"].isnull()]

    # Drop columns, make 
    df_processed.drop(columns=["Height", "Weight", "BMI"], inplace=True)
    # Remove the from col names too
    continuous_col_names = [col for col in continuous_col_names if col not in ["Height", "Weight", "BMI"]]

    print("Rows before filtering:", len(df))
    print("Rows after filtering: ", len(df_processed))

    df = df_processed.copy()

    if mean_imputation:
        print("Imputing missing values with mean...")
        # In the continuous columns, replace missing values with the mean
        for col in continuous_col_names:
            df[col] = df[col].fillna(df[col].mean())
    else:
        print("Imputing missing values normal distribution sampling...")
        # In the continuous columns, replace missing values with random samples from the same column
        for col in continuous_col_names:
            # Get the mean and standard deviation of the column
            mean = df[col].mean()
            std = df[col].std()
            # Generate random samples from a normal distribution with the same mean and std
            random_samples = np.random.normal(mean, std, size=df[col].isnull().sum())
            # Fill the missing values with the random samples
            df.loc[df[col].isnull(), col] = random_samples

    # LAB EVENTS
    # Convert the 'hadm_id' column to int64 in df_lab_events
    df_lab_events['hadm_id'] = df_lab_events['hadm_id'].astype('int64')

    print("Merging lab events with main dataset...")
    # Merge the two dataframes on the 'hadm_id' column
    df = pd.merge(df, df_lab_events, on='hadm_id', how='left')

    from sklearn.preprocessing import OrdinalEncoder
    # Apply ordinal encoding
    encoder = OrdinalEncoder(dtype=int, encoded_missing_value=-1)
    df[['gender']] = encoder.fit_transform(df[['gender']])
    df[['insurance']] = encoder.fit_transform(df[['insurance']])
    df[['race']] = encoder.fit_transform(df[['race']])

    # Fill NANs with False (which should only be present in the categorical columns at this point)
    df.fillna(False, inplace=True)

    # Sort the data based on "icd_code"
    df = df.sort_values(by='icd_code')
    df.drop(columns=['hadm_id', 'subject_id'], inplace=True)

    # rename icd_code to identity
    df.rename(columns={"icd_code": "identity"}, inplace=True)
    
    df.info()

    # Save the processed file in the same folder as input_path
    output_dir = os.path.dirname(input_path)
    output_path = os.path.join(output_dir, "processed_data.pkl")
    df.to_pickle(output_path)
    print(f"Processed data saved to: {output_path}")
    return output_path


def extract_features_and_split(processed_path, desried_num_unique_classes, desired_num_features, print_classification_reports=True):
    """
    Basic example function to extract features from the processed MIMIC dataset using Random Forest feature importance.
    This function loads the processed dataset, trains a Random Forest model, and extracts feature importances.
    Select features based on a threshold and saves the selected features to a new DataFrame.
    We then split the dataset into public and private datasets.
    Private dataset is desired_num_unique_classes of all samples with more than 1000 samples. Selects the classes with the most samples.
    Public dataset is the rest of ALL data.
    
    Args:
        processed_path (str): Path to the processed MIMIC dataset.
        desried_num_unique_classes (int): Desired number of unique classes for the private dataset.
        
    Returns:
        None
        Saves the public and private datasets to pickle files to same folder as processed_path.
    """
def extract_features_and_split(processed_path, desried_num_unique_classes, desired_num_features, print_classification_reports=True):
    """
    Loads the processed dataset, trains RF to rank features, keeps the top-N features,
    then splits into PRIVATE (top-K frequent classes) and PUBLIC (the rest).
    Writes two mapping files:
      - mapping.yaml           : top-K -> {0..K-1}
      - mapping_public.yaml    : rest   -> {K..C-1}
    Saves public_df.pkl and private_df.pkl next to processed_path.
    """

    # Load the processed dataset
    df = pd.read_pickle(processed_path)

    # Change some column data types (ensure RF can handle them; categories must already be numeric-encoded)
    df['num_procedures'] = df['num_procedures'].astype('int64')
    df['num_medications'] = df['num_medications'].astype('int64')
    df['race'] = df['race'].astype('category')
    df['insurance'] = df['insurance'].astype('category')
    df['gender'] = df['gender'].astype('category')

    # Initial stats
    init_len = len(df)
    init_unique = df["identity"].nunique()

    # Keep a copy of ALL data (for counts/splitting later)
    df_copy = df.copy()

    # For RF feature ranking, filter to identities with >1000 samples
    df = df.groupby("identity").filter(lambda x: len(x) > 1000)

    max_unique_classes = df["identity"].nunique()
    print(f"Number of unique classes with more than 1000 samples: {max_unique_classes}")

    # 90/10 stratified split for RF
    df_train, df_val = train_test_split(df, test_size=0.1, stratify=df['identity'], random_state=42)
    df_train = df_train.reset_index(drop=True)
    df_val = df_val.reset_index(drop=True)

    # Encode target for RF
    le = LabelEncoder()
    df_train['identity'] = le.fit_transform(df_train['identity'])
    df_val['identity'] = le.transform(df_val['identity'])

    # Split features/target
    X_train = df_train.drop(columns=['identity'])
    y_train = df_train['identity']
    X_val = df_val.drop(columns=['identity'])
    y_val = df_val['identity']

    # Train RF for feature importances
    print("Training Random Forest model...")
    rf_model = RandomForestClassifier(n_estimators=100, verbose=True, random_state=42, n_jobs=-1)
    rf_model.fit(X_train, y_train)
    y_pred = rf_model.predict(X_val)

    if print_classification_reports:
        print("Classification Report:")
        print(classification_report(y_val, y_pred, target_names=[str(cls) for cls in le.classes_]))

    # Rank features, keep top-N
    importances = rf_model.feature_importances_
    feature_importances = pd.DataFrame(importances, index=X_train.columns, columns=["importance"]).sort_values("importance", ascending=False)
    feature_importances = feature_importances.head(desired_num_features)
    print("Number of features selected: ", len(feature_importances))

    # Reduce BOTH datasets to the selected features + identity
    df_important_features = df[feature_importances.index].copy()
    df_copy_important_features = df_copy[feature_importances.index].copy()
    df_important_features["identity"] = df["identity"].astype('int64')
    df_copy_important_features["identity"] = df_copy["identity"].astype('int64')

    # We’ll proceed with the ALL-data reduced copy for splitting
    df_copy = df_copy_important_features

    # Keeping identities with occurance > 1000
    df_copy = df_important_features

    classes = sorted(df_copy['identity'].unique().tolist())
    mapping = {int(old): i+1 for i, old in enumerate(classes)}  # 1..C

    # (optional) save mapping
    output_dir = Path(processed_path).parent
    with open(output_dir / "mapping_1_based.yaml", "w") as f:
        yaml.safe_dump(mapping, f)

    # Apply the mapping in-place
    df_copy['identity'] = df_copy['identity'].map(mapping).astype('int64')


    PRIVATE_FRAC = 0.15   # 15% private, 85% public
    SEED = 42             # for reproducibility

    private_df = (
        df_copy
        .groupby('identity', group_keys=False)
        .apply(lambda g: g.sample(frac=PRIVATE_FRAC, random_state=SEED))
    )
    public_df = df_copy.drop(private_df.index)
    total = len(private_df) + len(public_df)
    print("Classes kept (after >1000 filter):", df_copy['identity'].nunique())
    print("Private rows:", len(private_df), "| Public rows:", len(public_df), "| Total:", total)
    print("Private ratio:", f"{len(private_df)/total:.2%}")
    print("Public ratio:",  f"{len(public_df)/total:.2%}")

    # 5) (optional) save splits
    private_df.to_pickle(output_dir / "private_df.pkl")
    public_df.to_pickle(output_dir / "public_df.pkl")

    


    # # Count classes over ALL data (reduced)
    # class_counts = df_copy['identity'].value_counts()
    # all_classes = class_counts.index.tolist()
    # C = len(all_classes)
    # K = int(desried_num_unique_classes)
    # if K <= 0 or K > C:
    #     raise ValueError(f"desired_num_unique_classes must be in 1..{C}, got {K}")

    # topK = all_classes[:K]
    # rest = all_classes[K:]

    # # Build mappings (two files)
    # mapping_private = {int(old): int(new) for new, old in enumerate(topK)}           # 0..K-1
    # mapping_public  = {int(old): int(new) for new, old in enumerate(rest, start=K)}  # K..C-1

    # # Save mappings
    # output_dir = Path(processed_path).parent
    # mapping_path_private = output_dir / "mapping.yaml"
    # mapping_path_public  = output_dir / "mapping_public.yaml"
    # with open(mapping_path_private, "w") as f:
    #     yaml.safe_dump(mapping_private, f)
    # with open(mapping_path_public, "w") as f:
    #     yaml.safe_dump(mapping_public, f)
    # print(f"Saved top-{K} mapping to {mapping_path_private}")
    # print(f"Saved remaining {C-K} mapping to {mapping_path_public}")

    # # Apply FULL mapping (now every class maps to an int; no NaNs)
    # mapping_full = {**mapping_private, **mapping_public}
    # df_copy["identity_mapped"] = df_copy["identity"].map(mapping_full).astype("int64")  # <-- FIXED: map from 'identity'

    # # Split
    # private_df = df_copy[df_copy["identity_mapped"] < K].copy()
    # public_df  = df_copy[df_copy["identity_mapped"] >= K].copy()

    # # Diagnostics (use the right columns)
    # print("Public dataset shape: ", public_df.shape)
    # print("Private dataset shape: ", private_df.shape)
    # print("Number of rows in public_df: ", len(public_df))
    # print("Number of rows in private_df: ", len(private_df))
    # print("Sum of rows in public_df and private_df: ", len(public_df) + len(private_df))
    # print("Initial len: ", init_len)

    # # Original labels in public; mapped labels in private
    # print("Number of unique ORIGINAL classes in public_df: ", public_df["identity"].nunique())
    # print("Number of unique MAPPED classes in private_df: ", private_df["identity_mapped"].nunique())
    # print("Initial unique: ", init_unique)
    # print("Public dataset info:")
    # public_df.info()
    # print("Private dataset info:")
    # private_df.info()

    # # Save the splits
    # private_output_path = os.path.join(output_dir, "private_df.pkl")
    # public_output_path = os.path.join(output_dir, "public_df.pkl")
    # public_df.to_pickle(public_output_path)
    # private_df.to_pickle(private_output_path)
