# standard modules
import math
import os
import time

# third-party
import numpy as np
import scipy
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


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
    train_losses, _ = train_model(
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

    test_predictions, test_targets = test_model(model, test_loader, device)
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

            # Create sequences
            X, y, indices = create_multidim_sequences(sequence_scaled, seq_length, horizon)

            if len(X) > 0:  # Only add if we have sequences
                processed_data[data_type].append({"X": X, "y": y, "indices": indices})
                print(f"{data_type} set: Added {len(X)} sequences")
            else:
                print(f"Warning: No sequences created for {data_type} data")

    return processed_data


def test_model(model, test_loader, device):
    # Evaluate on test set
    model.eval()
    test_predictions = []
    test_targets = []

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            test_predictions.append(outputs.cpu().numpy())
            test_targets.append(targets.cpu().numpy())

    # Concatenate batches
    test_predictions = np.vstack(test_predictions)
    test_targets = np.vstack(test_targets)

    return test_predictions, test_targets


def train_model(model, train_loader, criterion, optimizer, device, num_epochs, val_loader=None, print_every=10):
    train_losses = []
    val_losses = []

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
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for inputs, targets in val_loader:
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    val_loss += loss.item()

            val_loss /= len(val_loader)
            val_losses.append(val_loss)
            message += f", Val Loss: {val_loss:.4f}"
        else:
            val_losses = None

        if (epoch + 1) % print_every == 0:
            print(message)

    return train_losses, val_losses


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
