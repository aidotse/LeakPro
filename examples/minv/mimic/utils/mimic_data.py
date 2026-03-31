import pandas as pd


path = "data/physionet.org/files/mimiciv/3.1/"

# Load required tables
admissions = pd.read_csv(path + "hosp/admissions.csv")
patients = pd.read_csv(path + "hosp/patients.csv")
diagnoses = pd.read_csv(path + "hosp/diagnoses_icd.csv")
procedures = pd.read_csv(path + "hosp/procedures_icd.csv")
services = pd.read_csv(path + "hosp/services.csv")
prescriptions = pd.read_csv(path + "hosp/prescriptions.csv")
omr = pd.read_csv(path + "hosp/omr.csv")

#print(omr.head())

# Merge patient and admission data
df = admissions.merge(patients, on="subject_id", how="left")

# Convert admission time to datetime format
df["admittime"] = pd.to_datetime(df["admittime"])
df["dischtime"] = pd.to_datetime(df["dischtime"])

# Compute length of stay (LOS)
df["length_of_stay"] = (df["dischtime"] - df["admittime"]).dt.days

# Keep relevant columns
df = df[['subject_id', 'hadm_id', 'gender', 'insurance', 'race', 'length_of_stay']]

# Add **diagnosis labels**
diagnoses = diagnoses.groupby("hadm_id").first().reset_index()
df = df.merge(diagnoses[['hadm_id', 'icd_code']], on="hadm_id", how="left")
df['icd_code'] = df['icd_code'].astype(str)

# ================================
# ADDITIONAL FEATURE EXTRACTION
# ================================

## **1. Add Procedures as Features**
procedures_count = procedures.groupby("hadm_id")['icd_code'].count().reset_index()
procedures_count.rename(columns={'icd_code': 'num_procedures'}, inplace=True)
df = df.merge(procedures_count, on="hadm_id", how="left")
df['num_procedures'].fillna(0, inplace=True)  # Fill missing values with 0

## **2. Add Hospital Service Type**
df = df.merge(services[['hadm_id', 'curr_service']], on="hadm_id", how="left")
df['curr_service'].fillna('UNKNOWN', inplace=True)  # Replace missing values
df = pd.get_dummies(df, columns=['curr_service'])  # One-hot encode service type

## **3. Add Prescriptions Count**
prescriptions_count = prescriptions.groupby("hadm_id")['drug'].count().reset_index()
prescriptions_count.rename(columns={'drug': 'num_medications'}, inplace=True)
df = df.merge(prescriptions_count, on="hadm_id", how="left")
df['num_medications'].fillna(0, inplace=True)  # Fill missing values with 0


# Step 1: Split "Blood Pressure" into systolic and diastolic
def process_blood_pressure(value):
    if isinstance(value, str) and '/' in value:
        # Split the blood pressure value into systolic and diastolic
        try:
            systolic, diastolic = value.split('/')
            return float(systolic), float(diastolic)
        except ValueError:
            return None, None  # In case of any error during conversion
    return None, None

# Apply the function to rows where the result_name is "Blood Pressure"
omr['systolic'], omr['diastolic'] = zip(*omr['result_value'].where(omr['result_name'] == 'Blood Pressure').apply(process_blood_pressure))

# Step 2: Convert other numeric values (like weight, BMI, etc.) to numeric
omr['result_value'] = pd.to_numeric(omr['result_value'], errors='coerce')

# Step 3: Pivot the table, aggregating `result_value` where needed, using mean for numeric data
omr_processed = omr.pivot_table(index='subject_id', columns='result_name', values='result_value', aggfunc='mean')

# Handle 'systolic' and 'diastolic' as separate features for 'Blood Pressure'
omr_processed['systolic'] = omr['systolic'].mean()  # or use other aggregation (e.g., median)
omr_processed['diastolic'] = omr['diastolic'].mean()  # or use other aggregation

# Reset index to merge with the main dataframe
omr_processed.reset_index(inplace=True)

# Show the processed omr data
#print(omr_processed)

# Proceed with merging into the main dataframe
df = df.merge(omr_processed, on="subject_id", how="left")

# Fill missing values with median or another strategy
#df.fillna(df.median(), inplace=True)

# Only keep medications given BEFORE diagnosis time (optional)
#df = df[df['event_time'] < df['admittime']]

# Convert medications into binary features (one-hot encoding)
#df = pd.get_dummies(df, columns=['medication'])

# Aggregate medication features per patient
#df = df.groupby('hadm_id').max().reset_index()

emar = pd.read_csv(path + "hosp/emar.csv", usecols=['hadm_id', 'medication'])

print(emar.head())

top_10_meds = emar['medication'].value_counts().nlargest(100).index.tolist()

# Step 2: Filter only rows with top 10 medications
emar_filtered = emar[emar['medication'].isin(top_10_meds)]

# Step 3: One-hot encode the selected medications
emar_filtered['medication'] = emar_filtered['medication'].astype(str)  # Ensure it's a string
emar_encoded = pd.get_dummies(emar_filtered, columns=['medication'], prefix='med')

# Step 4: Aggregate by `hadm_id` to get one row per admission (max to indicate if med was given)
emar_encoded = emar_encoded.groupby('hadm_id').max().reset_index()

# Step 5: Merge with the main dataframe
df = df.merge(emar_encoded, on='hadm_id', how='left')

# ================================
# DATA PREPROCESSING
# ================================

# One-hot encoding for categorical variables
#categorical_features = ['gender', 'insurance', 'race']
#df = pd.get_dummies(df, columns=categorical_features)

# Label encode ICD codes (Multi-class classification)
df['icd_code'] = df['icd_code'].astype('category').cat.codes

# Extract features and labels
X = df.drop(columns=['hadm_id', 'icd_code'])  # Features
y = df['icd_code']  # Labels

print(X.head())
print(X.shape)
print(y.head())
# print unique values in y
print(y.unique().shape)

# pickle df
df.to_pickle("data/df.pkl")