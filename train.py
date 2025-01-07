from wandb.integration.sb3 import WandbCallback
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
import gymnasium as gym
import argparse
from clearml import Task
import wandb
import typing_extensions as TypeIs
import tensorflow
from tensorflow.python.checkpoint import checkpoint
import os


os.environ['WANDB_API_KEY'] = '6fca5bbd6d2177bec8096793bd4845c408625667'
from ot2_env_wrapper import OT2Env
# Load the API key for wandb
run = wandb.init(project="RL_task11", sync_tensorboard=True)

task = Task.init(
    project_name="Mentor Group D/Group 3/StijnvanderPas",
    task_name="First model",
    
)

# Set base docker image and queue
task.set_base_docker("deanis/2023y2b-rl:latest")
task.execute_remotely(queue_name="default")
# After running the setup script we can upload this
from ot2_env_wrapper import OT2Env
env = OT2Env()  

# Argument parser setup
parser = argparse.ArgumentParser()
parser.add_argument("--learning_rate", type=float, default=0.0001, help="Learning rate for the PPO model")
parser.add_argument("--batch_size", type=int, default=128, help="Batch size for the PPO model")
parser.add_argument("--n_steps", type=int, default=2048, help="Number of steps per PPO update")
parser.add_argument("--n_epochs", type=int, default=1, help="Number of epochs for PPO optimization")
args, unknown = parser.parse_known_args()  # Handles Jupyter environments gracefully

# Initialize the PPO model
model = PPO(
    "MlpPolicy",
    env,
    verbose=2,
    learning_rate=args.learning_rate,
    batch_size=args.batch_size,
    n_steps=args.n_steps,
    n_epochs=args.n_epochs,
    tensorboard_log=f"runs/{run.id}",
)

# Create path to save models
model_dir = f"models/{run.id}"
os.makedirs(model_dir, exist_ok=True)

# Create wandb callback
wandb_callback = WandbCallback(
    model_save_freq=100000, 
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
    
