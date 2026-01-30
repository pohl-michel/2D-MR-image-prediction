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
    configs_all_hrz = []
    if not config.get("horizons"):
        raise ValueError("This population transformer training app requires horizons to be specified in the config.")

    for horizon in config["horizons"]:
        # Adding the config for the current horizon
        tmp_config = copy.deepcopy(config)  # because config has sub-dicionaries, just being cautious here
        tmp_config["horizon"] = horizon
        configs_all_hrz.append(tmp_config)  # configs_all_hrz is made of deep copies, whereas config is unaffected

    if args.tuning and config.get("param_grid") is not None:
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
            tmp_config.update(best_params)

    else:
        print("Skipping hyperparameter tuning as it was not requested or no parameter grid is defined in the config.")
        # Attempting to retrieve optuna results json files
        for horizon_idx, horizon in enumerate(config["horizons"]):
            try:
                crt_hrz_config = configs_all_hrz[horizon_idx]
                most_recent_optuna_results_file = tf.get_most_recent_file(
                    folder_path=crt_hrz_config["paths"]["output_dir"],
                    pattern=f"optuna_results_h{horizon}_*_*.json",
                    regex_pattern=rf"optuna_results_h{horizon}_(\d{{8}})_(\d{{6}})\.json",
                )
                with open(most_recent_optuna_results_file, "r") as f:
                    optuna_results = json.load(f)
                    crt_hrz_config.update(optuna_results["best_params"])

            except FileNotFoundError:
                print(f"Warning: No optuna results found for horizon {horizon}. Using default parameters.")

    # Training the population transformer model for each horizon with the best parameters (if tuning was done)
    for horizon_idx, horizon in enumerate(config["horizons"]):
        print(f"Training population transformer model for horizon: {horizon}")
        crt_hrz_config = configs_all_hrz[horizon_idx]

        # Setting SHL field for inference in Matlab - could be improved later
        crt_hrz_config["SHL"] = crt_hrz_config["seq_length"]

        # Preprocessing data and splitting the data into pairs of input and target values
        standardized_data, _ = tf.preprocess_data(
            resampled_data, horizon, crt_hrz_config["seq_length"], crt_hrz_config["training_fraction"]
        )

        # Getting pytorch data loaders
        data_loaders = tf.get_population_data_loaders(standardized_data, crt_hrz_config["batch_size"])

        horizon_results = tf.train_multiple_models(
            config=crt_hrz_config,
            device=device,
            data_loaders=data_loaders,  # Rk: the best SHL might be different from that in the default config
            save_dir=crt_hrz_config["paths"]["output_dir"],
            print_every=config["print_every"],
        )

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
