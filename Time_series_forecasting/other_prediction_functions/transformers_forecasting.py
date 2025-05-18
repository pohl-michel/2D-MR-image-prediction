# standard modules
from datetime import datetime
import json
import math
import os
import time

# third-party
import numpy as np
import scipy
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, ConcatDataset


DATA_FILENAME = "data.mat"


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

    train_loader = get_data_loader(X_train, y_train, pred_par["batch_size"], shuffle=True)
    test_loader = get_data_loader(X_test, y_test, pred_par["batch_size"], shuffle=False)
    n_features = X_train.shape[-1]  # Number of features (dimensions)

    device = get_device(pred_par["selected_device"])
    print(f"Using device: {device}")

    model = MultiDimTimeSeriesTransformer(
        input_dim=n_features,  # Number of input features (dimensions)
        output_dim=n_features,  # Number of output features (dimensions)
        seq_length=pred_par["SHL"],  # SHL
        d_model=pred_par["d_model"],  # embedding dimension - should be divisible by nhead
        nhead=pred_par["nhead"],  # number of attention heads
        num_layers=pred_par["num_layers"],  # number of transformer layers
        dim_feedforward=pred_par["dim_feedforward"],  # feedforward dimension inside encoder
        final_layer_dim=pred_par["final_layer_dim"],  # feedforward dimension of final layer
        dropout=pred_par["dropout"],  # dropout rate
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=pred_par["learn_rate"])

    start_time = time.monotonic()
    train_losses, _, _ = train_model(
        model,
        train_loader,
        criterion,
        optimizer,
        device,
        num_epochs=pred_par["num_epochs"],
        val_loader=None,
        print_every=pred_par["print_every"],
    )
    end_time = time.monotonic()
    elapsed_time = end_time - start_time
    avg_pred_time = elapsed_time / len(train_loader)

    test_predictions, test_targets, _ = eval_model(model, test_loader, device)
    # I can get the loss separately in Matlab as in svr_pred

    return test_predictions, avg_pred_time


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

    # Extract parameters from config
    input_dim = config.get("n_features", 1)
    output_dim = config.get("n_features", 1)
    seq_length = config.get("seq_length", 24)
    d_model = config.get("d_model", 16)
    nhead = config.get("nhead", 2)
    num_layers = config.get("num_layers", 2)
    dim_feedforward = config.get("dim_feedforward", 64)
    dropout = config.get("dropout", 0.1)
    final_layer_dim = config.get("final_layer_dim", None)

    # Create model
    return MultiDimTimeSeriesTransformer(
        input_dim=input_dim,
        output_dim=output_dim,
        seq_length=seq_length,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
        final_layer_dim=final_layer_dim,
    )

def train_multiple_models(
    config,
    device,
    data_loaders,
    horizon,
    n_models,
    num_epochs,
    learning_rate,
    early_stop_patience,
    save_dir,
    print_every=10,
):
    """
    Train multiple horizon-specific models to account for stochasticity

    Args:
        horizon: Prediction horizon (steps ahead)
        n_models: Number of models to train
        num_epochs: Maximum number of training epochs
        learning_rate: Learning rate for optimization
        early_stopping_patience: Patience for early stopping
        print_every: Print training progress every N epochs
        save_dir: Directory to save models

    Returns:
        List of training results for each model
    """

    # Update config with horizon
    config["horizon"] = horizon

    # Prepare directory
    horizon_dir = os.path.join(save_dir, f"horizon_{horizon}")
    os.makedirs(horizon_dir, exist_ok=True)

    # Train multiple models
    all_results = []

    for model_idx in range(n_models):
        print(f"\n{'='*50}")
        print(f"Training model {model_idx+1}/{n_models} for horizon={horizon}")
        print(f"{'='*50}")

        # Set different random seed for each model
        seed = config.get("seed", 42) + model_idx
        torch.manual_seed(seed)
        np.random.seed(seed)

        # Initialize new model
        model = init_model(config).to(device)

        # Set up training
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        train_losses, val_losses, best_model_state = train_model(
            model,
            data_loaders["train"],
            criterion,
            optimizer,
            device,
            num_epochs,
            val_loader=data_loaders["val"],
            print_every=print_every,
            early_stop_patience=early_stop_patience,
        )

        # Restore best model
        if best_model_state is not None:
            model.load_state_dict(best_model_state)

        # Evaluate on test set if available
        if data_loaders["test"] is not None:
            _, _, test_loss = eval_model(model, data_loaders["test"], device, criterion, return_predictions=False)
            print(f"Test Loss for Model {model_idx+1}: {test_loss:.4f}")

        # Save this model
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = os.path.join(horizon_dir, f"transformer_h{horizon}_model{model_idx+1}_{timestamp}.pt")
        config_path = os.path.join(horizon_dir, f"transformer_h{horizon}_model{model_idx+1}_{timestamp}_config.json")

        # Save model and config
        torch.save(best_model_state, model_path)

        training_history = {
            "train_losses": train_losses,
            "val_losses": val_losses if data_loaders["val"] is not None else None,
        }

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

        print(f"Model {model_idx+1} saved to {model_path}")

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


def eval_model(model, test_loader, device, criterion=None, return_predictions=True):
    # Evaluate on test set
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

        if (epoch + 1) % print_every == 0:
            print(message)

    return train_losses, val_losses, best_model_state


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
