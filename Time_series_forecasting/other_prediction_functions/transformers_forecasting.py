# standard modules
from datetime import datetime
import json
import math
import os
import time
import glob
import re

# third-party
import numpy as np
import optuna
from optuna.trial import Trial
import scipy
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, ConcatDataset


DATA_FILENAME = "data.mat"
MODEL_DIR = "d. RNN variables (temp)"


def generate_multidimensional_sinusoids(n_samples=1000, n_features=3, noise_level=0.2):
    """Generate multiple noisy sinusoid time series with different frequencies and amplitudes"""
    t = np.arange(n_samples)

    # Initialize arrays for clean and noisy signals
    clean_signals = np.zeros((n_features, n_samples))
    noisy_signals = np.zeros((n_features, n_samples))

    # Generate different sinusoids for each feature
    for i in range(n_features):
        # Vary frequency and amplitude for each dimension
        freq = 0.01 + (i * 0.01)  # Different frequency for each dimension
        amplitude = 1.0 + (i * 0.5)  # Different amplitude for each dimension
        phase = i * (np.pi / 4)  # Different phase for each dimension

        # Create the clean sinusoid
        clean_signals[i] = amplitude * np.sin(2 * np.pi * freq * t + phase)

        # Add Gaussian noise
        noise = np.random.normal(0, noise_level, n_samples)
        noisy_signals[i] = clean_signals[i] + noise

    return noisy_signals, clean_signals


def create_multidim_sequences(data, seq_length, horizon):
    """
    Create input/output sequences for multidimensional time series prediction with variable horizon

    Parameters:
        data: numpy array of shape [n_features, n_samples]
        seq_length: length of input sequence
        horizon: how many steps ahead to predict

    Returns:
        xs: input sequences [n_samples, seq_length, n_features]
        ys: target values [n_samples, n_features]
        indices: indices of the targets in the original time series
    """
    xs, ys, indices = [], [], []
    n_dims, n_samples = data.shape

    for i in range(n_samples - seq_length - horizon + 1):
        # Extract sequence of shape (seq_length, n_dims)
        x = data[:, i : i + seq_length].transpose()  # Transpose to get (seq_length, n_dims)

        # Extract target values (horizon steps ahead for all dimensions)
        target_idx = i + seq_length + horizon - 1
        y = data[:, target_idx]

        xs.append(x)
        ys.append(y)
        indices.append(target_idx)

    return np.array(xs), np.array(ys), np.array(indices)


def get_device(selected_device: str):
    """Function selecting the appropriate device"""

    if selected_device == "cuda":
        if not torch.cuda.is_available():
            print("CUDA is not available. Using CPU instead.")
            return torch.device("cpu")
        else:
            return torch.device("cuda")
    else:
        return torch.device("cpu")


def train_and_predict(pred_par, X_train, y_train, X_test, y_test):
    """Function to call inside Matlab, doing training and prediction on a test (or validation) set, in a sequence-wise
    manner. X_train, y_train, X_test, y_test are assumed to come from the same sequence, that was standardized
    beforehand (so this function does not handle standardization)."""

    # Check if we need to reshape inputs
    if len(X_train.shape) == 2:  # If MATLAB dropped the feature / dimension (unidimensional signal)
        # Reshape to add the feature dimension back
        X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
        X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
        y_train = y_train.reshape(y_train.shape[0], 1)
        y_test = y_test.reshape(y_test.shape[0], 1)

    # Check that the signal history length in the config file corresponds to the signal history length in X_test
    (nb_samples, shl, n_features) = X_test.shape
    if pred_par["SHL"] != shl:
        raise ValueError(f"Signal history length in config ({pred_par['SHL']}) does not match X_test ({shl}).")

    train_loader = get_data_loader(X_train, y_train, pred_par["batch_size"], shuffle=True)
    test_loader = get_data_loader(X_test, y_test, pred_par["batch_size"], shuffle=False)

    device = get_device(pred_par["selected_device"])
    print(f"Using device: {device}")

    # Create model
    transformer_config = {  # Extract relevant parameters for the transformer model from pred_par
        key: val
        for key, val in pred_par.items()
        if key in ["d_model", "nhead", "num_layers", "dim_feedforward", "dropout", "final_layer_dim"]
    }
    transformer_config.update({"n_features": n_features, "seq_length": pred_par["SHL"]})

    # Initialize model
    model = init_model(transformer_config)

    # Sending the model to the appropriate device
    model.to(device)

    train_losses, _, _ = train_model(
        model,
        train_loader,
        criterion=nn.MSELoss(),
        optimizer=torch.optim.Adam(model.parameters(), lr=pred_par["learn_rate"]),
        device=device,
        num_epochs=pred_par["num_epochs"],
        val_loader=None,
        print_every=pred_par["print_every"],
    )

    start_time = time.monotonic()
    # I can get the loss separately in Matlab as in svr_pred
    test_predictions, test_targets, _ = eval_model(model, test_loader, device)
    end_time = time.monotonic()
    elapsed_time = end_time - start_time
    avg_pred_time = elapsed_time / len(train_loader)  # only training time... could take inference into account...

    return test_predictions, avg_pred_time


def population_model_predict(pred_par, X_test, run_idx):
    """Perform prediction with a previously trained population model, on a test set"""

    # Load model configuration file - not needed here because all the relevant parameters should be in pred_par ...
    # model_dir = os.path.join(MODEL_DIR, f"horizon_{pred_par['horizon']}")  # directory containing results
    # config_path = get_most_recent_config(model_dir, pred_par["horizon"], run_idx)
    # with open(config_path, "r") as f:
    #     config = json.load(f)

    # Check if we need to reshape inputs
    if len(X_test.shape) == 2:  # If MATLAB dropped the feature / dimension (unidimensional signal)
        X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

    # Check that the signal history length in the config file corresponds to the signal history length in X_test
    (nb_samples, shl, n_features) = X_test.shape
    if pred_par["seq_length"] != shl:
        raise ValueError(f"Signal history length in config ({pred_par['seq_length']}) does not match X_test ({shl}).")
    if pred_par["n_features"] != n_features:
        raise ValueError(f"Nb of features in config ({pred_par['n_features']}) does not match X_test ({n_features}).")

    # Initialize model
    model = init_model(pred_par)

    # Get the device,
    device = get_device(pred_par["selected_device"])
    print(f"Using device: {device}")

    # Load saved model file
    model_path = pred_par["config_path"].removesuffix("_config.json") + ".pt"

    # Load model weights
    model.load_state_dict(torch.load(model_path, map_location=device))

    # Sending the model to the appropriate device
    model.to(device)

    # Load the scaler file - where is my scaler file :( - think about it later - I need to save it in notebook
    # Rk: actually no, I can just use sequence-wise scaling during inference, that should work

    # Do the forecasting - the test data is set as arbitrary because it is not used for the prediction
    test_loader = get_data_loader(X_test, np.zeros((X_test.shape[0], 1)), pred_par["batch_size"], shuffle=False)

    start_time = time.monotonic()
    test_predictions, test_targets, _ = eval_model(model, test_loader, device)
    end_time = time.monotonic()
    elapsed_time = end_time - start_time
    avg_pred_time = elapsed_time / len(test_loader)

    # Rescale back to original scale - do later
    # R: no, cf comment above.

    return test_predictions, avg_pred_time


def get_most_recent_config(folder_path, horizon, run_idx):
    """
    Args:
        folder_path (str): Path to the folder containing config files
        run_idx (int, optional): Specific run index to filter by (e.g., 1 for model1)
    Rk: same as get_most_recent_model_config() in Matlab - to refactor later so that I have only one function...
    """
    # Pattern to match the files
    pattern = os.path.join(folder_path, f"transformer_h{horizon}_model{run_idx}_*_*_config.json")
    files = glob.glob(pattern)

    if not files:
        raise FileNotFoundError("No config files found in the specified folder")

    most_recent_file = None
    most_recent_datetime = datetime.min

    # Regex pattern to extract date and time
    regex_pattern = rf"transformer_h{horizon}+_model{run_idx}_(\d{{8}})_(\d{{6}})_config\.json"

    for file_path in files:
        filename = os.path.basename(file_path)
        match = re.search(regex_pattern, filename)

        if match:
            date_str = match.group(1)  # YYYYMMDD
            time_str = match.group(2)  # HHMMSS

            # Parse the datetime
            file_datetime = datetime.strptime(f"{date_str}_{time_str}", "%Y%m%d_%H%M%S")

            if file_datetime > most_recent_datetime:
                most_recent_datetime = file_datetime
                most_recent_file = file_path

    if most_recent_file is None:
        raise ValueError("No valid files found with the expected naming pattern")

    return most_recent_file


def get_data_loader(X, y, batch_size, shuffle: bool):
    """Create a DataLoader for the dataset"""

    # Convert to PyTorch tensors
    X_tensor = torch.FloatTensor(X)  # Shape: [batch, seq_len, features]
    y_tensor = torch.FloatTensor(y)

    # Create data loader
    tensor_dataset = TensorDataset(X_tensor, y_tensor)
    data_loader = DataLoader(tensor_dataset, batch_size=batch_size, shuffle=shuffle)

    return data_loader


def load_data_core(folders_list: list[str], data_type: str = ""):
    """Load data from a list of folders"""

    data_list = []
    metadata_list = []

    for folder in folders_list:
        file_path = os.path.join(folder, DATA_FILENAME)
        if not os.path.exists(file_path):
            print(f"Warning: File not found - {file_path}")
            continue

        print(f"Loading {data_type} data from {folder}...")
        if file_path.endswith(".mat"):
            # Load MATLAB file
            mat_data = scipy.io.loadmat(file_path)
            signal_data = mat_data.get("org_data")
            if signal_data is None:
                raise KeyError(f"Key 'org_data' not found in {file_path}")

        else:
            print(f"Unsupported file format: {file_path}")
            continue

        # Store data and metadata
        data_list.append(signal_data)
        metadata_list.append(
            {
                "folder": folder,
                "file_path": file_path,
                "n_features": signal_data.shape[0],
                "n_samples": signal_data.shape[1],
            }
        )
    return data_list, metadata_list


def load_dev_test_data(dev_folders, test_folders):
    """
    Load data from separate development and test folder lists

    Args:
        dev_folders: List of folders for development set (training + validation)
        test_folders: List of folders for test set
        data_filename: Name of the data file in each folder (default: 'data.mat')

    Returns:
        Dictionary containing the loaded datasets
    """
    # Validate inputs
    if not dev_folders:
        raise ValueError("Development folders list is empty")

    dev_data, dev_metadata = load_data_core(dev_folders, data_type="development")
    test_data, test_metadata = load_data_core(test_folders, data_type="test")

    # Verify all signals have the same number of features
    n_features_list = [meta["n_features"] for meta in dev_metadata + test_metadata]
    if len(set(n_features_list)) > 1:
        raise ValueError(f"Inconsistent number of features across files: {n_features_list}")

    n_features = n_features_list[0]

    # Store the data
    raw_data = {
        "dev_data": dev_data,
        "dev_metadata": dev_metadata,
        "test_data": test_data,
        "test_metadata": test_metadata,
    }

    return raw_data, n_features


def resample_data_poly(dev_data, dev_freq, test_freq):
    """
    Resample development data to match the frequency of test data using polyphase filtering

    Args:
        dev_freq: Sampling frequency of development data (Hz)
        test_freq: Sampling frequency of test data (Hz)
    """

    # Calculate resampling ratio as fraction
    gcd = np.gcd(int(dev_freq * 1000), int(test_freq * 1000))
    up = int((test_freq * 1000) / gcd)  # Upsampling factor
    down = int((dev_freq * 1000) / gcd)  # Downsampling factor

    print(f"Resampling with ratio {dev_freq}/{test_freq} = {down}/{up}")

    # Resample development data
    resampled_dev_data = []

    for i, patient_data in enumerate(dev_data):
        n_features, n_samples = patient_data.shape

        # Calculate approximate new number of samples (for reporting only)
        approx_new_samples = int(n_samples * (test_freq / dev_freq))

        # Apply resample_poly to entire patient data at once
        # This works when all features should be resampled the same way
        resampled_signal = scipy.signal.resample_poly(patient_data, up, down, axis=1)

        resampled_dev_data.append(resampled_signal)

        print(
            f"Resampled dev signal {i+1}: {n_samples} → {resampled_signal.shape[1]} "
            f"samples (expected ~{approx_new_samples})"
        )

    return resampled_dev_data


def preprocess_data(data_in, horizon, seq_length, training_fraction):
    """
    Preprocess the loaded data (resampled or raw) for training and testing, using standardization,
    and splits sequences into pairs of examples (input and target values).

    Args:
        data: Dictionary containing the loaded datasets:
            - dev_data: List of numpy arrays for development data
            - test_data: List of numpy arrays for test data
        horizon: Prediction horizon (steps ahead)
        seq_length: Length of input sequences
        training_fraction: Fraction of development data to use for training
            (the rest is used for validation)

    Returns:
        Dictionary containing preprocessed datasets
    """

    # Split development data into training and validation, and use test data as is
    data = {"train": [], "val": [], "test": data_in["test_data"]}

    for patient_data in data_in["dev_data"]:
        n_samples = patient_data.shape[1]
        train_end = int(training_fraction * n_samples)

        data["train"].append(patient_data[:, :train_end])
        data["val"].append(patient_data[:, train_end:])

    # Fit scaler on training data
    all_train_data = np.hstack(data["train"]).T  # Shape: [total_samples, n_features]
    scaler = StandardScaler()
    scaler.fit(all_train_data)

    # Process each dataset
    processed_data = {"train": [], "val": [], "test": []}

    for data_type, input_data in data.items():
        for sequence in input_data:

            # Scale the data
            sequence_scaled = scaler.transform(sequence.T).T

            # Create sequences: input [n_samples, seq_length, n_features], target [n_samples, n_features]
            X, y, indices = create_multidim_sequences(sequence_scaled, seq_length, horizon)

            if len(X) > 0:  # Only add if we have sequences
                processed_data[data_type].append({"X": X, "y": y, "indices": indices})
                print(f"{data_type} set: Added {len(X)} sequences")
            else:
                print(f"Warning: No sequences created for {data_type} data")

    return processed_data, scaler


def get_population_data_loaders(processed_data, batch_size):
    """Create data loaders for training and evaluation"""

    # Create combined datasets for each split
    data_loaders = {}

    for dataset_name in ["train", "val", "test"]:
        # Skip if no data
        if not processed_data[dataset_name]:
            data_loaders[dataset_name] = None
            continue

        # Create tensor datasets
        tensor_datasets = []

        for patient_data in processed_data[dataset_name]:
            X = torch.FloatTensor(patient_data["X"])
            y = torch.FloatTensor(patient_data["y"])
            tensor_datasets.append(TensorDataset(X, y))

        # Combine datasets
        combined_dataset = ConcatDataset(tensor_datasets)

        # Create data loader
        data_loaders[dataset_name] = DataLoader(
            combined_dataset, batch_size=batch_size, shuffle=(dataset_name == "train")
        )

        print(f"{dataset_name.capitalize()} loader: {len(combined_dataset)} sequences")

    return data_loaders


def init_model(config):
    """Initialize the transformer model with the given configuration.
    That's useful when we need to initialize different models with different seeds"""

    # Create model
    end_params = {
        key: config[key]
        for key in ["seq_length", "d_model", "nhead", "num_layers", "dim_feedforward", "dropout", "final_layer_dim"]
    }
    return MultiDimTimeSeriesTransformer(input_dim=config["n_features"], output_dim=config["n_features"], **end_params)


def hyperparameter_tuning(
    config: dict,
    param_grid: dict,
    resampled_data,
    device,
    n_trials=None,
    pruner=None,
    sampler=None,
    n_jobs=1,
    study_name="transformer_optimization",
    save_dir="tmp",
):
    """
    Perform hyperparameter tuning using Optuna

    Args:
        n_trials: Number of optimization trials
        study_name: Name for the Optuna study
        save_dir: Directory to save results

    Returns:
        Best parameters found
    """
    # Create directories if they don't exist
    os.makedirs(save_dir, exist_ok=True)

    if n_trials is None:
        # Calculate the exact number of combinations in the grid
        total_combinations = 1
        for values in param_grid.values():
            total_combinations *= len(values)

        # If n_trials is not specified, set it to the grid size
        n_trials = total_combinations
        print(f"Setting n_trials={n_trials} to match grid size")

    # Define the objective function
    def objective(trial: Trial):

        # Define hyperparameters to tune
        params = {param: trial.suggest_categorical(param, values) for param, values in param_grid.items()}

        # Add to config (note: this will overwrite existing values)
        temp_config = config.copy()
        temp_config.update(params)

        print(f"Generating sequences with length {temp_config['seq_length']}...")

        standardized_data, scaler = preprocess_data(
            resampled_data, temp_config["horizon"], temp_config["seq_length"], temp_config["training_fraction"]
        )
        data_loaders = get_population_data_loaders(standardized_data, batch_size=temp_config["batch_size"])

        # early_stop_patience set to None because we don't use early stopping in the tuning
        all_results = train_multiple_models(
            temp_config,
            device,
            data_loaders,
            early_stop_patience=None,
            save_dir=None,
            print_every=temp_config["print_every"],
            trial=None,
        )

        # Extract the final validation losses from the results and compute their mean
        final_val_losses = [result["training_history"]["val_losses"][-1] for result in all_results]
        mean_val_loss = np.mean(final_val_losses)

        return mean_val_loss

    # Create the study
    sampler = optuna.samplers.GridSampler(param_grid)
    study = optuna.create_study(study_name=study_name, direction="minimize", pruner=pruner, sampler=sampler)

    # Run optimization
    study.optimize(objective, n_trials=n_trials, n_jobs=n_jobs)

    # Get the best parameters
    print(f"Best trial: {study.best_trial.number}")
    print(f"Best value: {study.best_trial.value}")
    print("Best hyperparameters:")

    for param_name, param_value in study.best_trial.params.items():
        print(f"    {param_name}: {param_value}")

    # Save study results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = os.path.join(save_dir, f"optuna_results_{timestamp}.json")

    # Convert study results to a serializable format
    results = {
        "best_params": study.best_trial.params,
        "best_value": study.best_trial.value,
        "best_trial": study.best_trial.number,
        "datetime": timestamp,
        "n_trials": n_trials,
        "all_trials": [
            {
                "number": t.number,
                "value": t.value if t.value is not None else None,
                "params": t.params,
                "state": t.state.name,
            }
            for t in study.trials
        ],
    }

    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Study results saved to {results_path}")

    return study.best_trial.params


def train_multiple_models(
    config,
    device,
    data_loaders,
    early_stop_patience,
    save_dir=None,
    print_every=10,
    trial: Trial = None,
):
    """
    Train multiple horizon-specific models to account for stochasticity

    Args:
        early_stopping_patience: Patience for early stopping
        print_every: Print training progress every N epochs
        save_dir: Directory to save models

    Returns:
        List of training results for each model
    """
    work_config = config.copy()  # Copy the config to avoid modifying the original

    # Train multiple models
    all_results = []

    if trial is not None:
        if data_loaders["val"] is None:
            raise ValueError("Validation data loader is required for Optuna trials")

        if early_stop_patience is not None and trial is not None:  # using early stopping could bias the trials
            print("Warning: Early stopping potentially interacting with Optuna trials.")

        data_loaders["test"] = None  # No test data during hyperparameter optimization

    # List containing the validation losses of each model at the end of training
    val_losses_all_models = []

    for model_idx in range(work_config["nb_runs"]):
        print(f"\n{'='*50}")
        print(f"Training model {model_idx+1}/{work_config['nb_runs']} for horizon={work_config['horizon']}")
        print(f"{'='*50}")

        # Set different random seed for each model
        seed = work_config.get("seed", 42) + model_idx
        torch.manual_seed(seed)
        np.random.seed(seed)

        # Initialize new model
        model = init_model(work_config).to(device)

        # Set up training
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=work_config["learning_rate"])

        train_losses, val_losses, best_model_state = train_model(
            model,
            data_loaders["train"],
            criterion,
            optimizer,
            device,
            work_config["num_epochs"],
            val_loader=data_loaders["val"],
            print_every=print_every,
            early_stop_patience=early_stop_patience,
            data_augmentation_config=work_config.get("data_augmentation", None),
        )

        # Updating list of validation losses for all models at the end of training
        val_losses_all_models.append(val_losses[-1])

        if trial is not None:  # this is in case we are doing hyperparameter optimization

            # Report intermediate values
            trial.report(np.mean(val_losses_all_models), model_idx)

            # Handle pruning based on the intermediate value
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

        elif data_loaders["test"] is not None:
            # Evaluate on test set if available, except during hyperparameter optimization
            _, _, test_loss = eval_model(model, data_loaders["test"], device, criterion, return_predictions=False)
            print(f"Test Loss for Model {model_idx+1}: {test_loss:.4f}")

        # Restore best model
        if best_model_state is not None:
            model.load_state_dict(best_model_state)

        training_history = {
            "train_losses": train_losses,
            "val_losses": val_losses if data_loaders["val"] is not None else None,
        }

        model_path, config_path = None, None
        if save_dir is not None:
            model_path, config_path = save_model(
                best_model_state, work_config, model_idx, seed, training_history, save_dir
            )

        # Store results - scaler not saved with np.save(scaler_path, scaler_dict) because they are common to all models
        all_results.append(
            {
                "model_idx": model_idx + 1,
                "model_path": model_path,
                "config_path": config_path,
                "training_history": training_history,
                "test_loss": test_loss if data_loaders["test"] is not None else None,
            }
        )

    return all_results


def save_model(best_model_state, config, model_idx, seed, training_history, save_dir):
    """Save the model and its configuration.

    Args:
        best_model_state (dict): The state of the best model.
        config (dict): The configuration used for training.
        model_idx (int): The index of the model.
        seed (int): The random seed used for training.
        timestamp (str): The timestamp when the model was trained.
        training_history (dict): The training history of the model.
        save_dir (str): The directory where the model should be saved.
    """

    # Create directory for this horizon if it doesn't exist
    horizon = config["horizon"]
    horizon_dir = os.path.join(save_dir, f"horizon_{horizon}")
    os.makedirs(horizon_dir, exist_ok=True)

    # Save this model
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = os.path.join(horizon_dir, f"transformer_h{horizon}_model{model_idx+1}_{timestamp}.pt")
    config_path = os.path.join(horizon_dir, f"transformer_h{horizon}_model{model_idx+1}_{timestamp}_config.json")
    # scaler_path = os.path.join(horizon_dir, f"transformer_h{horizon}_model{model_idx+1}_{timestamp}_scaler.npy")

    # Save model and config
    torch.save(best_model_state, model_path)

    # Add training history to config
    model_info = {
        "config": config,
        "model_idx": model_idx + 1,
        "horizon": horizon,
        "seed": seed,
        "timestamp": timestamp,
        "training_history": training_history,
    }

    with open(config_path, "w") as f:
        json.dump(model_info, f, indent=2)

    # # Save scaler
    # scaler_dict = {"mean": scaler.mean_, "var": scaler.var_, "scale": scaler.scale_}
    # np.save(scaler_path, scaler_dict)

    print(f"Model {model_idx+1} saved to {model_path}")

    return model_path, config_path


def eval_model(model, test_loader, device, criterion=None, return_predictions=True):
    # Evaluate on test set

    # Set model to evaluation mode
    model.eval()

    test_predictions = []
    test_targets = []
    total_loss = 0.0

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            if return_predictions:
                test_predictions.append(outputs.cpu().numpy())
                test_targets.append(targets.cpu().numpy())

            if criterion is not None:
                loss = criterion(outputs, targets)
                total_loss += loss.item()

    # Concatenate batches
    if return_predictions:
        test_predictions = np.vstack(test_predictions)
        test_targets = np.vstack(test_targets)

    avg_loss = total_loss / len(test_loader)

    return test_predictions, test_targets, avg_loss


def train_model(
    model,
    train_loader,
    criterion,
    optimizer,
    device,
    num_epochs,
    val_loader=None,
    print_every=10,
    early_stop_patience=None,
    data_augmentation_config=None,
):
    train_losses = []
    val_losses = []

    best_model_state = None
    if early_stop_patience is not None:  # We perform stopping
        best_val_loss = float("inf")
        patience_counter = 0

    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0.0
        for inputs, targets in train_loader:
            # print(f"input shape: {inputs.shape}, target shape: {targets.shape}")
            # Move data to the appropriate device

            if data_augmentation_config is not None:
                inputs, targets = time_series_augmentation_suite(inputs, targets, data_augmentation_config)

            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        message = f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}"

        # Validation
        if val_loader is not None:
            _, _, val_loss = eval_model(model, val_loader, device, criterion, return_predictions=False)
            val_losses.append(val_loss)
            message += f", Val Loss: {val_loss:.4f}"

            if early_stop_patience is not None:
                # Check for improvement
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_model_state = model.state_dict().copy()
                    patience_counter = 0
                else:
                    patience_counter += 1

                # Early stopping
                if patience_counter >= early_stop_patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break

        else:
            val_losses = None

        if (print_every is not None) and ((epoch + 1) % print_every == 0):
            print(message)

    return train_losses, val_losses, best_model_state


def time_series_augmentation_suite(batch_data, batch_targets, config: dict):
    """
    Comprehensive augmentation suite for time series data. Note: we are not adding noise because the PCA breathing
    signals are already noisy. Both the probabilities of applying each transformation and the transformation parameters
    need to be provided, otherwise the augmentation will not be applied.

    Args:
        batch_data: Tensor of shape [batch_size, seq_length, n_features]
        batch_targets: Tensor of shape [batch_size, n_features]
        config: Dictionary of augmentation parameters

    Returns:
        Augmented batch data and targets
    """
    if config is None:
        raise ValueError("No data augmentation configuration provided. Please provide a valid config dictionary.")

    augmented_data = batch_data.clone()
    augmented_targets = batch_targets.clone()

    batch_size, seq_length, n_features = batch_data.shape

    for i in range(batch_size):

        # Scaling (amplitude variation)
        if (
            (config.get("scaling_range", None) is not None)
            and (config.get("scaling_prob", None) is not None)
            and (np.random.rand() < config["scaling_prob"])
        ):
            for j in range(n_features):
                scale = np.random.uniform(*config["scaling_range"])
                augmented_data[i, :, j] *= scale
                augmented_targets[i, j] *= scale

        # Feature Permutation
        if (config.get("permutation_prob", None) is not None) and (np.random.rand() < config["permutation_prob"]):
            perm = torch.randperm(n_features)
            augmented_data[i, :, :] = augmented_data[i, :, perm]
            augmented_targets[i, :] = augmented_targets[i, perm]

        # Calculate feature-wise amplitude for proportional transformations
        amplitudes = []
        for j in range(n_features):
            # Estimate amplitude as 95th percentile - 5th percentile for robustness
            q95 = torch.quantile(augmented_data[i, :, j], 0.95)
            q05 = torch.quantile(augmented_data[i, :, j], 0.05)
            amplitude = q95 - q05
            amplitudes.append(amplitude.item())

        # Add Baseline Bias (constant offset proportional to amplitude)
        if (
            (config.get("bias_prob", None) is not None)
            and (config.get("max_bias_factor", None) is not None)
            and (np.random.rand() < config["bias_prob"])
        ):
            for j in range(n_features):
                # Generate random bias proportional to feature amplitude
                max_bias = amplitudes[j] * config["max_bias_factor"]
                bias = np.random.uniform(-max_bias, max_bias)

                # Apply constant bias to entire feature sequence
                augmented_data[i, :, j] += bias

                # Apply same bias to the target
                augmented_targets[i, j] += bias

        # Add Random Drift (linear slope)
        if (
            (config.get("drift_prob", None) is not None)
            and (config.get("max_drift_factor", None) is not None)
            and (np.random.rand() < config["drift_prob"])
        ):
            for j in range(n_features):
                # Random slope proportional to feature amplitude
                max_change = amplitudes[j] * config["max_drift_factor"]
                slope = np.random.uniform(-max_change, max_change)

                # Create linear trend along time dimension
                time_steps = np.arange(seq_length) / seq_length
                drift = torch.FloatTensor(time_steps) * slope

                # Apply drift to this feature
                augmented_data[i, :, j] += drift

                # Adjust target based on final drift value
                final_drift = drift[-1].item()
                augmented_targets[i, j] += final_drift

    return augmented_data, augmented_targets


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, max_len):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        # x shape: [batch_size, seq_len, d_model]
        return x + self.pe[:, : x.size(1), :]


class MultiDimTimeSeriesTransformer(nn.Module):

    DIM_FF_PROPORTIONALITY_FACTOR = 2  # Proportionality factor for feedforward dimension

    def __init__(
        self,
        input_dim,
        output_dim,
        seq_length,
        d_model,
        nhead,
        num_layers,
        dropout,
        dim_feedforward=None,
        final_layer_dim=None,
    ):

        # Setting the feedforward dimension as proportional to d_model if not provided
        if dim_feedforward in [None, 0]:
            dim_feedforward = int(d_model * self.DIM_FF_PROPORTIONALITY_FACTOR)

        # Setting the final network layer dimension to the geometric mean of its input and output, if not provided
        if final_layer_dim in [None, 0]:
            final_layer_dim = int(np.sqrt(seq_length * d_model * output_dim))

        super(MultiDimTimeSeriesTransformer, self).__init__()
        self.input_projection = nn.Linear(input_dim, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_len=seq_length)  # Setting pe "max_len" param to SHL

        # Create a transformer encoder layer with explicit dropout parameter
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, batch_first=True
        )

        # Stack multiple encoder layers
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Feed Forward Network for final prediction (outputs all dimensions simultaneously)
        self.ffn = nn.Sequential(
            nn.Linear(d_model * seq_length, final_layer_dim),  # Use all sequence outputs
            nn.ReLU(),
            nn.Linear(final_layer_dim, output_dim),  # Output all dimensions
        )

    def forward(self, src):
        # src shape: [batch_size, seq_len, input_dim]

        # Project to d_model dimensions
        src = self.input_projection(src)

        # Add positional encoding
        src = self.positional_encoding(src)

        # Pass through transformer encoder
        transformer_output = self.transformer_encoder(src)

        # Flatten the output for the FFN
        batch_size = transformer_output.size(0)
        flattened = transformer_output.reshape(batch_size, -1)

        # Pass through feed-forward network for final prediction of all dimensions
        output = self.ffn(flattened)

        return output
