# CNN Model for Host Tree Classification using NAIP Imagery and Environmental Variables
# Thomas Lake, July 2025

# Imports
import os

conda_env = r"C:\Users\talake2\AppData\Local\anaconda3\envs\naip_ailanthus_env"
os.environ["GDAL_DATA"] = os.path.join(conda_env, "Library", "share", "gdal")
os.environ["PROJ_LIB"] = os.path.join(conda_env, "Library", "share", "proj")
os.environ["PATH"] += os.pathsep + os.path.join(conda_env, "Library", "bin")

import time  # noqa: E402
import logging # noqa: E402
import argparse # noqa: E402
import json # noqa: E402
import pandas as pd # noqa: E402
import torch # noqa: E402
from torch.utils.data import DataLoader # noqa: E402
from model import HostImageryClimateModel, HostImageryOnlyModel, HostClimateOnlyModel, HostImageryClimateTopoModel, HostTopoOnlyModel # noqa: E402
from datasets import HostNAIPDataset # noqa: E402
from transforms import RandomAugment4Band # noqa: E402
from eval_utils import test_model, plot_accuracies, plot_losses, map_model_errors # noqa: E402
from train_utils import fit # noqa: E402
from logging_utils import setup_logging # noqa: E402

import wandb  # noqa: E402

def main():
    # Argument Parser
    parser = argparse.ArgumentParser(description="Train Host NAIP Imagery and Environmental Variables Model for Host Classification")
    parser.add_argument('--config', type=str, required=True, help='Path to the configuration JSON file')
    args = parser.parse_args()

    # Load Config
    with open(args.config, 'r') as f:
        config = json.load(f)

    # Create experiment subdirectory
    experiment_name = config.get('experiment')
    experiment_dir = os.path.join(config['output_dir'], experiment_name)
    os.makedirs(experiment_dir, exist_ok=True)

    # Create experiment-specific checkpoint directory
    checkpoint_dir = os.path.join(experiment_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Logging
    log_path = os.path.join(experiment_dir, f"{experiment_name}.log")
    setup_logging(log_path)

    # Init W&B and override config with W&B sweep values for hyperparameter search
    wandb.init(name = experiment_name, entity="talake2-ncsu", project="naip_climate_classification")

    if wandb.run: # True if running in a W&B sweep
        sweep_run_id = wandb.run.name
        config["experiment"] += f"_{sweep_run_id}"
        config['epcohs'] = wandb.config.get('epochs', config['epochs'])
        config['batch_size'] = wandb.config.get('batch_size', config['batch_size'])
        config['learning_rate'] = wandb.config.get('learning_rate', config['learning_rate'])
        config['dropout'] = wandb.config.get('dropout', config.get('dropout', 0.25)) # Default dropout of 0.25 if not specified

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Image Transformations
    image_transform = RandomAugment4Band(
    rotation_degrees=config['rotation_degrees'],
    hflip_prob=config['hflip_prob'],
    vflip_prob=config['vflip_prob']
    )

    # Dataset
    env_vars = config['env_features'] # List of environmental variables (WorldClim, GHM, DEM, etc.) from config
    input_mode = config.get('input_mode', 'baseline')
    train_ds = HostNAIPDataset(config['csv_path'], config['image_dir'], 'train', env_vars, transform=image_transform, input_mode=input_mode)
    val_ds = HostNAIPDataset(config['csv_path'], config['image_dir'], 'val', env_vars, input_mode=input_mode)
    test_ds = HostNAIPDataset(config['csv_path'], config['image_dir'], 'test', env_vars, input_mode=input_mode) # Return lat/lon for mapping errors

    # Dataloaders
    train_dl = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True, num_workers=4)
    val_dl = DataLoader(val_ds, batch_size=config['batch_size'], shuffle=False, num_workers=4)
    test_dl = DataLoader(test_ds, batch_size=config['batch_size'], shuffle=False, num_workers=4)

    # Model
    dropout = config.get('dropout', 0.25)  # Default dropout if not specified
    print("Using dropout:", dropout)

    model_type = config.get("model_type", "image_climate") # Default to combined NAIP + Env Model
    print(model_type)
    if model_type in {"image_climate", "naip_climate"}:
        model = HostImageryClimateModel(num_env_features=len(env_vars), dropout=dropout).to(device)
    elif model_type == "image_only":
        model = HostImageryOnlyModel(dropout=dropout).to(device)
    elif model_type == "climate_only":
        model = HostClimateOnlyModel(num_env_features=len(env_vars), dropout=dropout).to(device)
    elif model_type == "topo_only":
        model = HostTopoOnlyModel(dropout=dropout).to(device)
    elif model_type in {"naip_topo_climate", "image_topo_climate"}:
        model = HostImageryClimateTopoModel(num_env_features=len(env_vars), dropout=dropout).to(device)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])

    # Log the Model Arguments 
    logging.info(f"Model Arguments: {json.dumps(config, indent=4)}")
    wandb.config.update({"dropout": dropout}, allow_val_change=True)

    start_time = time.time()

    # Fit Model ( Training Loop )
    history = fit(
        config['epochs'],
        config['learning_rate'],
        model,
        train_dl,
        val_dl,
        optimizer,
        checkpoint_dir,
        lr_patience=config['lr_patience'],
        es_patience=config['es_patience']
    )

    elapsed_time = time.time() - start_time
    logging.info(f"Training completed in {elapsed_time:.2f} seconds")

    # Save Model Training History    
    output_csv = os.path.join(experiment_dir, "training_history.csv")
    print("Saving training history to:", output_csv)
    pd.DataFrame(history).to_csv(output_csv, index=False)
    logging.info(f"Training history saved to {output_csv}")

    # Plot Accuracies and Losses
    plot_accuracies(history, experiment_dir)
    plot_losses(history, experiment_dir)
                                
    # Model Evaluation and Confusion Matrix
    test_model(model, test_dl, device, experiment_dir)

    # Map Model Errors (Plot Testing Points by Error Type)
    map_model_errors(model, test_dl, device, experiment_dir)

if __name__ == "__main__":
    main()
