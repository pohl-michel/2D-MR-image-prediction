import argparse
import copy
import json
import os
import sys

from . import transformers_forecasting as tf

CONFIG_DIR = "other_prediction_functions"


def main(argv):
    parser = argparse.ArgumentParser(description="Train population transformer model")
    parser.add_argument(
        "--config",
        help="path to JSON config file",
        default="pop_transformer_training_config_test.json",  # the test config is the default config
        nargs=1,
        type=str,
    )
    parser.add_argument("--tuning", help="Enable hyperparameter tuning", action="store_true", default=False)
    parser.add_argument("--njobs", help="Number of jobs for hyperparameter tuning", default=1, type=int)
    parser.add_argument("--gpu", help="Use GPU for training", action="store_true", default=False)
    args = parser.parse_args(argv)

    print("✅ Training population transformer model...")
    config_file = args.config[0] if isinstance(args.config, list) else args.config
    print(f"Using config file: {config_file}")

    with open(os.path.join(CONFIG_DIR, config_file), "r") as f:
        config = json.load(f)

    # Loading time series data
    folders = {}
    for folder_type in ["dev_folders", "test_folders"]:
        folders[folder_type] = [
            os.path.join(config["paths"]["input_sq_dir"], folder) for folder in config["paths"][folder_type]
        ]
    raw_data, n_features = tf.load_dev_test_data(**folders)

    # Setting the number of features in the config
    config["n_features"] = n_features

    # Resampling development set data so that its frequency matches the test set
    resampled_dev_data = tf.resample_data_poly(
        raw_data["dev_data"],
        source_freq=config["training_sq_sampling_rate_hz"],
        target_freq=config["test_sq_sampling_rate_hz"],
    )
    resampled_data = raw_data.copy()
    resampled_data["dev_data"] = resampled_dev_data

    # Get the device
    requested_device = "cuda" if args.gpu else "cpu"
    device = tf.get_device(requested_device)
    print(f"Using device: {device}")

    # Creating one config per horizon
    configs_all_hrz, data_loaders_all_hrz = [], []
    for horizon in config["horizons"]:
        # Adding the config for the current horizon
        tmp_config = copy.deepcopy(config)  # because config has sub-dicionaries, just being cautious here
        tmp_config["horizon"] = horizon
        configs_all_hrz.append(tmp_config)

        # Preprocessing data and splitting the data into pairs of input and target values
        standardized_data, _ = tf.preprocess_data(
            resampled_data, horizon, tmp_config["seq_length"], tmp_config["training_fraction"]
        )

        # Getting pytorch data loaders
        data_loaders = tf.get_population_data_loaders(standardized_data, batch_size=tmp_config["batch_size"])

        # Adding the data loaders for the current horizon
        data_loaders_all_hrz.append(data_loaders)

    if args.tuning and config.get("param_grid") is not None:
        if not config.get("horizons"):
            raise ValueError("Hyperparameter tuning requires horizons to be specified in the config.")

        for horizon_idx, horizon in enumerate(config["horizons"]):
            print(f"Running hyperparameter tuning for horizon: {horizon}")
            tmp_config = configs_all_hrz[horizon_idx]  # get the config for the current horizon

            # Copy the config, set the SHL field for inference in Matlab
            best_params = tf.hyperparameter_tuning(
                tmp_config,
                param_grid=tmp_config["param_grid"],  # not ideal as that's part of the config but that works for now
                data=resampled_data,
                device=device,
                n_jobs=args.njobs,
                n_trials=tmp_config.get("n_trials"),
                pruner=None,  # we don't use pruning in the current code, but we can add that later
                study_name=tmp_config.get("study_name", "transformer_optimization"),
                save_dir=tmp_config["paths"]["output_dir"],  # same comment as above concerning passing tmp config
            )
            print(f"Best parameters for horizon {horizon}: {best_params}")

            # Rk: I do not run grid search over the batch size or training fraction, but that may be accounted for below
            if tmp_config["seq_length"] != best_params.get("seq_length", tmp_config["seq_length"]):
                print(f"Updating sequence length from {tmp_config['seq_length']} to {best_params['seq_length']}")
                standardized_data, _ = tf.preprocess_data(
                    resampled_data, horizon, best_params["seq_length"], tmp_config["training_fraction"]
                )
                data_loaders = tf.get_population_data_loaders(standardized_data, tmp_config["batch_size"])
                data_loaders_all_hrz[horizon_idx] = data_loaders

            tmp_config.update(best_params)

    # Training the population transformer model for each horizon with the best parameters (if tuning was done)
    for horizon_idx, horizon in enumerate(config["horizons"]):
        print(f"Training population transformer model for horizon: {horizon}")
        config_crt_hrz = configs_all_hrz[horizon_idx]

        # Setting SHL field for inference in Matlab - could be improved later
        config_crt_hrz["SHL"] = config_crt_hrz["seq_length"]

        horizon_results = tf.train_multiple_models(
            config=config_crt_hrz,
            device=device,
            data_loaders=data_loaders_all_hrz[horizon_idx],
            save_dir=config_crt_hrz["paths"]["output_dir"],
            print_every=config["print_every"],
        )

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
