from wandb.integration.sb3 import WandbCallback
from stable_baselines3 import PPO
import os
import random
import wandb
import gc

# Ensure deterministic results for reproducibility (except for new seeds)
SEED = 11

# Disable GPU (if desired)
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# Path to save models
model_dir_base = "models"
os.makedirs(model_dir_base, exist_ok=True)

# Function to randomly generate parameters
def generate_parameters():
    learning_rate = random.uniform(1e-5, 1e-3)  # Random learning rate between 1e-5 and 1e-2
    batch_size = random.choice([64, 128, 256, 512])  # Random batch size from predefined options
    seed = random.randint(1, 10000)  # Random seed between 1 and 10,000
    return learning_rate, batch_size, seed

# Simulated number of iterations
total_iterations = 2000
iteration = 0

while iteration < total_iterations:
    iteration += 1

    # Generate new parameters
    learning_rate, batch_size, seed = generate_parameters()
    print(f"Starting iteration {iteration} with learning_rate={learning_rate}, batch_size={batch_size}, seed={seed}")

    # Start a new wandb run
    run = wandb.init(project="RL_task11", sync_tensorboard=True)

    # Log parameters to wandb
    wandb.config.update({
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "seed": seed,
        "iteration": iteration
    })

    from ot2_env_wrapper_updated_3 import OT2Env
    env = OT2Env()

    # Initialize the PPO model
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=learning_rate,
        batch_size=batch_size,
        n_steps=2048,  # Fixed value
        n_epochs=10,   # Fixed value
        tensorboard_log=f"runs/{run.id}",
        seed=seed  # Pass the seed to the model
    )

    # Create a unique directory for each iteration's model
    model_dir = os.path.join(model_dir_base, f"iteration_{iteration}")
    os.makedirs(model_dir, exist_ok=True)

    # Create wandb callback
    wandb_callback = WandbCallback(
        model_save_freq=500000,
        model_save_path=model_dir,
        verbose=2
    )

    # Train the model
    try:
        time_steps = 2500000  # Total training timesteps per iteration
        model.learn(
            total_timesteps=time_steps,
            callback=wandb_callback,
            progress_bar=True,
            reset_num_timesteps=False,
            tb_log_name=f"runs/{run.id}"
        )

        # Save the model after training
        model.save(f"{model_dir}/{time_steps * iteration}")
        print(f"Model saved at iteration {iteration}: {model_dir}/{time_steps * iteration}")

    except KeyboardInterrupt:
        print("Training interrupted by user.")
        break

    except Exception as e:
        print(f"An error occurred during iteration {iteration}: {e}")
        continue

    # Clean up resources
    model.env.close()  # Close the environment to free resources
    del model
    gc.collect()  # Force garbage collection to clear memory

    # Finish the current wandb run
    wandb.finish()

print("Training loop terminated.")
