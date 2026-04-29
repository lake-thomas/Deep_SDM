import wandb
import os
import yaml

with open('sweep.yaml', 'r') as file:
    sweep_config = yaml.safe_load(file)

# Create the sweep configuration
sweep_id = wandb.sweep(
    sweep_config,
    project='naip_climate_classification'
)

# Fun a funciton that calls main.py with the sweep config
def train():
    os.system("python main.py --config model_config.json")

# Launch the sweep
wandb.agent(
    sweep_id,
    function=train
)