from wandb.integration.sb3 import WandbCallback
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
import gymnasium as gym
import argparse
from clearml import Task
import wandb
import tensorflow
from tensorflow.python.checkpoint.checkpoint import Checkpoint
import os

# Ensure deterministic results with a fixed seed
SEED = 11

# Disable GPU (if desired)
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# Load the API key for wandb
os.environ['WANDB_API_KEY'] = '6fca5bbd6d2177bec8096793bd4845c408625667'
run = wandb.init(project="RL_task11", sync_tensorboard=True)

from ot2_env_wrapper_updated import OT2Env
env = OT2Env()

# Argument parser setup
parser = argparse.ArgumentParser()
parser.add_argument("--learning_rate", type=float, default=0.0003, help="Learning rate for the PPO model")
parser.add_argument("--batch_size", type=int, default=128, help="Batch size for the PPO model")
parser.add_argument("--n_steps", type=int, default=2048, help="Number of steps per PPO update")
parser.add_argument("--n_epochs", type=int, default=1, help="Number of epochs for PPO optimization")
args, unknown = parser.parse_known_args()  # Handles Jupyter environments gracefully

# Initialize the PPO model
model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    learning_rate=args.learning_rate,
    batch_size=args.batch_size,
    n_steps=args.n_steps,
    n_epochs=args.n_epochs,
    tensorboard_log=f"runs/{run.id}",
    seed=SEED  # Pass the seed to the model
)

# Log the seed to wandb
wandb.config.update({"seed": SEED})

# Create path to save models
model_dir = f"models/{run.id}"
os.makedirs(model_dir, exist_ok=True)

# Create wandb callback
wandb_callback = WandbCallback(
    model_save_freq=500000, 
    model_save_path=model_dir, 
    verbose=2
)

# Total training timesteps per iteration
time_steps = 3000000

# Training loop
for i in range(1):
    print(f"Starting learn iteration {i + 1}")
    
    # Train the model and log data
    model.learn(
        total_timesteps=time_steps,
        callback=wandb_callback,
        progress_bar=True,
        reset_num_timesteps=False,
        tb_log_name=f"runs/{run.id}"
    )
    print(f"Completed learn iteration {i + 1}")
    
    # Save the model after each iteration
    model.save(f"{model_dir}/{time_steps * (i + 1)}")
    print(f"Model saved at iteration {i + 1}: {model_dir}/{time_steps * (i + 1)}")

# Print a final message with seed information
print(f"Training complete. Seed used: {SEED}")