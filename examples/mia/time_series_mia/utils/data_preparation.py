import os, pickle, joblib, torch, random, numpy as np, pandas as pd

from scipy.io import loadmat
from sklearn.preprocessing import RobustScaler, MinMaxScaler, StandardScaler, FunctionTransformer
from torch import tensor, float32
from torch.utils.data import DataLoader, Dataset, TensorDataset, Subset
from sklearn.model_selection import train_test_split
from mne.io import read_raw_edf
from leakpro.utils.logger import logger

class IndividualizedDataset(Dataset):
    def __init__(self, data:tensor, targets:tensor, individual_indices:list[tuple[int,int]], scaler, stride, val_set=None, num_val_individuals=0):
        self.data = data
        self.targets = targets
        self.scaler = scaler
        self.stride = stride
        
        self.lookback = data.size(1)
        self.horizon = targets.size(1)
        self.num_variables = targets.size(2)

        self.individual_indices = individual_indices    # individual_indices[i] is a tuple [start_index, end_index) for individual i
        self.num_individuals = len(individual_indices)
        self.samples_per_individual = data.size(0) // self.num_individuals

        self.val_set = val_set  # either None or a TensorDataset
        self.num_val_individuals = num_val_individuals 

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx, ...], self.targets[idx, ...]
    
    @property
    def input_dim(self):
        return self.data.shape[-1]
    
    @property
    def output_dim(self):
        return self.targets.shape[-1]


def read_mat_data(path, file):
    file_path = os.path.join(path, file)
    return loadmat(file_path)['val'].T  # transpose to get shape (#timesteps, #variables)

def get_ECG_dataset(path, num_individuals, k_lead=12, **kwargs):
    """Get the ECG dataset."""

    timesteps = 5000   # number of timesteps in raw time series
    raw_data_path = os.path.join(path, 'ECG')
    individual_files = random.sample(os.listdir(raw_data_path), num_individuals)
    all_raw_time_series = np.array(list(filter(
        lambda ts: ts.shape[0] == timesteps, # keep time series with 5000 timesteps (only 52/10344 individuals don't satisfy this) 
        map(lambda f: read_mat_data(raw_data_path, f), individual_files) 
    )))
    all_raw_time_series = all_raw_time_series[..., :k_lead]

    return all_raw_time_series


def get_edf_time_series(edf_data, k_lead, num_time_steps, num_initial_time_steps_to_cut):
    time_series = edf_data.get_data()
    time_series = time_series.T # transpose to get sample dimension first
    
    start = num_initial_time_steps_to_cut
    end = start + num_time_steps
    if end > time_series.shape[0]:
        raise ValueError(f"Tried to sample steps [{start}, {end}) from time series of length {time_series.shape[0]}.")

    return time_series[start:end, :k_lead]  # select first num_timesteps of the k first variables after cutting the first num_initial_time_steps_to_cut

def get_EEG_dataset(path, num_individuals, k_lead=3, num_time_steps=30000, **kwargs):
    """Get the EEG dataset. Assuming subjects are placed in data/EEG with subfolders '000', '001', etc (as original structure).
        num_time_steps is the fixed number of steps to use from each individual; cutting the longer series and ignoring shorter ones
        default: 30000 steps (equivalent to 2 minutes when sampling at 250Hz)
    """
    # Fix sample frequency for all individual time series
    sample_frequency = 250

    # Cut the first minute of each time series to avoid initial artificial signals (possibly artifacts, test signals, calibrations, etc) present in some subjects
    num_intial_seconds_to_cut = 60
    num_initial_time_steps_to_cut = sample_frequency * num_intial_seconds_to_cut

    # Minimum time series length to match num_time_steps after cutting first minute
    min_n_times = num_time_steps + num_initial_time_steps_to_cut

    data_path = os.path.join(path, 'EEG')
    subjects = []
    for subdir in os.listdir(data_path):
        for subject in os.listdir(os.path.join(data_path, subdir)):
            subjects.append(f'{subdir}/{subject}')
    random.shuffle(subjects)   # shuffle subject list to ensure random selection order

    # Randomly select EEG recordings from unique subjects (at most one recording per subject)
    selected_individuals = []
    for subject in subjects:
        subject_tokens = [] # all valid tokens for current subject (potentially from multiple sessions)
        for session in os.listdir(os.path.join(data_path, subject)):
            dirs = os.listdir(os.path.join(data_path, f'{subject}/{session}'))
            if len(dirs) > 1:
                raise Exception(f'Expected single montage, but {subject}/{session} has {len(dirs)} montage definitions!')
            montage_definition = dirs[0]
            for token in os.listdir(os.path.join(data_path, f'{subject}/{session}/{montage_definition}')):
                file = os.path.join(data_path, f'{subject}/{session}/{montage_definition}/{token}')
                data = read_raw_edf(file, verbose=False)
                if data.info['sfreq'] == sample_frequency and data.n_times >= min_n_times:
                    subject_tokens.append(data) # keep only data sampled at a frequency of 250 Hz and long enough

        # Randomly select one valid recording from this subject (if any)
        if len(subject_tokens) > 0:
            selected_individuals.append(random.choice(subject_tokens))
        
        # Stop when the desired number of individuals is reached
        if len(selected_individuals) == num_individuals:
            break

    # Trim and return the selected time series
    if len(selected_individuals) < num_individuals:
        logger.warning(f"num_individuals = {num_individuals} but only found {len(selected_individuals)} with num_time_steps >= {num_time_steps} after cutting first minute (avoiding potential artificial signals). Proceeding with {len(selected_individuals)} individuals.")
    trimmed_selected_time_series = np.array([get_edf_time_series(individual, k_lead, num_time_steps, num_initial_time_steps_to_cut) for individual in selected_individuals])
    return trimmed_selected_time_series


def get_ELD_dataset(path, num_individuals, num_time_steps, **kwargs):
    """Get the ELD dataset."""

    df = pd.read_csv(os.path.join(path, "ELD", "LD2011_2014.txt"), delimiter=";", decimal=",")
    # Set a name for date column
    df.rename(columns={df.columns[0]: "Date"}, inplace=True)
    df["Date"] = pd.to_datetime(df["Date"], utc=True)
    
    # Resample to hourly frequency and sum the values
    df.set_index("Date", inplace=True)
    df = df.resample('h').sum()

    load_data = []

    for indiv in df.columns:
        load = df[indiv]

        load = np.array(load, dtype=np.float32)

        # Find the index of the first non-zero element
        first_non_zero = np.argmax(load != 0)
    
        # Find the index of the last non-zero element
        last_non_zero = len(load) - np.argmax(load[::-1] != 0) - 1
        
        # Slice the array to remove the first and last zeros
        load = load[first_non_zero:last_non_zero + 1]

        load_data.append(load)

    # Filter out shorter time series
    load_data = [load for load in load_data if len(load) >= num_time_steps]
    load_data = [load for load in load_data if np.mean(load) <= 2000 and np.mean(load) >= 200]
    assert num_individuals <= len(load_data), "Too few individuals in dataset (after filtering)"

    random.shuffle(load_data)
    load_data = load_data[:num_individuals]
    load_data = [load[:num_time_steps] for load in load_data]

    data = np.array(load_data)
    data = np.expand_dims(data, -1)
    return data


def get_LCL_dataset(path, num_individuals, **kwargs):
    """Get the LCL dataset."""
        
    df = pd.read_csv(os.path.join(path, "LCL", "daily_dataset.csv"))
    individuals = list(df["LCLid"].value_counts(sort=True, ascending=False)[:num_individuals].index)
    assert len(individuals) == num_individuals

    load_data = []
    time_data = []

    for indiv in individuals:
        indiv_df = df[df["LCLid"] == indiv]
        load = indiv_df["energy_mean"]
        time = indiv_df["day"]

        # Convert float32 and np.datetimes
        load = np.array(load, dtype=np.float32)
        time = np.array(time, dtype=np.datetime64)

        # Remove duplicates and replace by mean of the duplicates
        unique_time, _ = np.unique(time, return_index=True)
        avg_load = np.zeros_like(unique_time, dtype=np.float32)
        for i, t in enumerate(unique_time):
            mask = time == t
            avg_load[i] = np.mean(load[mask])  # Take average of duplicates
        time = unique_time
        load = avg_load

        # Generate the expected time range (30-minute intervals)
        start, end = time[0], time[-1]
        expected_time = np.arange(start, end + np.timedelta64(1, 'D'), np.timedelta64(1, 'D'))

        # Fill missing timestamps with NaN
        filled_load = np.full_like(expected_time, np.nan, dtype=np.float32)
        filled_load[np.isin(expected_time, time)] = load

        time = expected_time
        load = filled_load

        # Interpolate missing values linearly
        nan_mask = np.isnan(load)
        load[nan_mask] = np.interp(
            np.flatnonzero(nan_mask),
            np.flatnonzero(~nan_mask),
            load[~nan_mask]
        )

        load_data.append(load)
        time_data.append(time)

    seq_len = min(len(ts) for ts in load_data)
    time_data = [ts[:seq_len] for ts in time_data]
    load_data = [ts[:seq_len] for ts in load_data]
    time_data = np.expand_dims(np.array(time_data), -1)
    load_data = np.expand_dims(np.array(load_data), -1)
    return load_data


def dataset_matches_params(dataset_name, dataset, lookback, horizon, num_individuals, stride, scaling, val_fraction, **kwargs):
    """Check if a saved dataset matches the given parameters."""
    
    if dataset is None:
        return False
    
    # Get kwargs
    num_time_steps = kwargs.get("num_time_steps", None)
    k_lead = kwargs.get("k_lead", None)

    if k_lead is None or dataset_name in ["ELD", "LCL"]:    # note: ELD and LCL are univariate    
        matching_num_variables = True
    else:
        matching_num_variables = dataset.num_variables == k_lead

    if num_time_steps is None or dataset_name != "EEG": # note: only EEG needs this since it truncates series
        matching_num_time_steps = True 
    else:
        expected_num_time_steps = num_time_steps - lookback - horizon + 1
        matching_num_time_steps = (len(dataset) // dataset.num_individuals) == expected_num_time_steps

    if scaling.lower() == "none":
        matching_scaler = dataset.scaler.__class__.__name__ == "FunctionTransformer"
    else:
        matching_scaler = dataset.scaler.__class__.__name__.replace("Scaler", "").lower() == scaling.lower()

    # Check all parameters
    return (
        dataset.lookback == lookback and
        dataset.horizon == horizon and
        dataset.num_individuals + dataset.num_val_individuals == num_individuals and
        dataset.stride == stride and
        dataset.num_val_individuals == round(val_fraction * num_individuals) and
        matching_num_variables and
        matching_num_time_steps and
        matching_scaler
    )

def to_sequences(data, lookback, horizon, stride):
    """Create samples from data (raw time-series) by applying sliding window with specified lookback, horizon, and stride."""
    x, y = [], []
    timesteps = len(data)
    num_samples = timesteps - (lookback + horizon) + 1

    for t in range(0, num_samples, stride):
        x.append(data[t:t + lookback, :])
        y.append(data[t + lookback:t + lookback + horizon, :])
    return tensor(np.array(x), dtype=float32), tensor(np.array(y), dtype=float32)

def preprocess_dataset(dataset_name, path, lookback, horizon, num_individuals, stride, scaling, val_fraction, **kwargs):
    """Get and preprocess the dataset."""

    # Load dataset if already exists on path
    dataset = None

    # If all parameters matches, we're done; return it
    if dataset_matches_params(dataset_name, dataset, lookback, horizon, num_individuals, stride, scaling, val_fraction, **kwargs):
        return dataset

    # Else we need to construct the dataset
    # First get the raw time series of the dataset, with shape (individuals, num_time_steps)
    if dataset_name == "ECG":
        raw_data = get_ECG_dataset(path, num_individuals, **kwargs)
    elif dataset_name == "EEG":
        raw_data = get_EEG_dataset(path, num_individuals, **kwargs)
    elif dataset_name == "LCL":
        raw_data = get_LCL_dataset(path, num_individuals, **kwargs)
    elif dataset_name == "ELD":
        raw_data = get_ELD_dataset(path, num_individuals, **kwargs)
    else:
        raise NotImplementedError(f"Unknown dataset: {dataset_name}")
    
    # Scaling
    scaling = scaling.lower()
    scaler_dict = {'none': FunctionTransformer(), 
                   'standard': StandardScaler(), 
                   'minmax': MinMaxScaler(), 
                   'robust': RobustScaler()}
    if scaling not in scaler_dict.keys():
        raise NotImplementedError(f"Unknown scaler: {scaling}. Supported scalings are: {scaler_dict.keys()}")
    scaler = scaler_dict[scaling]
    data_concatenated = np.concatenate(raw_data)
    data_scaled = scaler.fit_transform(data_concatenated)

    # Reshape to include individual dimension again (this is OK since all time series have equal length)
    data_scaled = data_scaled.reshape(raw_data.shape)

    # Set aside the validation set and keep out of the audit set 
    val_size = round(val_fraction * num_individuals)
    if val_size > 0:
        train_test_individuals, val_individuals = train_test_split(np.arange(num_individuals), test_size=val_size)
    else:
        train_test_individuals, val_individuals = np.arange(num_individuals), []

    x = []  # lists to store samples for all individuals
    y = []
    x_val = []
    y_val = []
    for i, time_series in enumerate(data_scaled):
        # Create sequences separately for each individual
        xi, yi = to_sequences(time_series, lookback, horizon, stride)
        if i in train_test_individuals:
            x.append(xi)
            y.append(yi)
        else: # val_individuals
            x_val.append(xi)
            y_val.append(yi)

    # Keep track of sample indices for each individual time series
    num_samples_per_individual = len(x[0])
    individual_indices = [(0 + num_samples_per_individual*i, num_samples_per_individual*(i+1)) for i in range(len(x))]

    # Construct validation set
    if len(x_val) > 0:
        x_val, y_val = torch.cat(x_val, dim=0), torch.cat(y_val, dim=0)
        val_set = TensorDataset(x_val, y_val)
    else:
        val_set = None

    # Concatenate samples and save dataset
    x, y = torch.cat(x, dim=0), torch.cat(y, dim=0)
    dataset = IndividualizedDataset(x, y, individual_indices, scaler, stride, val_set, len(val_individuals))
    with open(f"{path}/{dataset_name}.pkl", "wb") as file:
        pickle.dump(dataset, file)
        print(f"Save data to {path}/{dataset_name}.pkl") 

    return dataset


def get_dataloaders(dataset: IndividualizedDataset, train_fraction=0.5, test_fraction=0.3, batch_size=128):

    tot_num_individuals = dataset.num_individuals + dataset.num_val_individuals
    train_size = int(train_fraction * tot_num_individuals)
    test_size = int(test_fraction * tot_num_individuals)
    train_individuals, test_individuals = train_test_split(np.arange(dataset.num_individuals), train_size=train_size, test_size=test_size)
        
    train_ranges = [dataset.individual_indices[i] for i in train_individuals]
    test_ranges = [dataset.individual_indices[i] for i in test_individuals]

    train_indices = np.concatenate([np.arange(start, stop) for (start, stop) in train_ranges], axis=0)
    test_indices = np.concatenate([np.arange(start, stop) for (start, stop) in test_ranges], axis=0)

    train_subset = Subset(dataset, train_indices) 
    test_subset = Subset(dataset, test_indices)

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset.val_set, batch_size=batch_size, shuffle=False) if dataset.val_set else None
    test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader