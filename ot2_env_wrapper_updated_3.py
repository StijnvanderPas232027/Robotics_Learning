import gymnasium as gym
from gymnasium import spaces
import numpy as np
from sim_class import Simulation

# Create the class
class OT2Env(gym.Env):
    def __init__(self, render=False, max_steps=1000):
        super(OT2Env, self).__init__()
        self.render = render
        self.max_steps = max_steps

        # Create the simulation environment
        self.sim = Simulation(num_agents=1, render=self.render)

        # Set the maximum values according to the working environment.
        self.x_min, self.x_max = -0.187, 0.2531
        self.y_min, self.y_max = -0.1705, 0.2195
        self.z_min, self.z_max = 0.1195, 0.2895
        # Define action and observation spaces
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=np.array([self.x_min, self.y_min, self.z_min, -self.x_max, -self.y_max, -self.z_max], dtype=np.float32),
            high=np.array([self.x_max, self.y_max, self.z_max, self.x_max, self.y_max, self.z_max], dtype=np.float32),
            dtype=np.float32
        )

        # Keep track of the step amount
        self.steps = 0
        self.previous_distance = None

    def reset(self, seed=None):
        # Set a seed if it was not set yet
        if seed is not None:
            np.random.seed(seed)

        # Randomise the goal position
        x = np.random.uniform(self.x_min, self.x_max)
        y = np.random.uniform(self.y_min, self.y_max)
        z = np.random.uniform(self.z_min, self.z_max)
        # Set a random goal position
        self.goal_position = np.array([x, y, z])
        # Call reset function
        observation = self.sim.reset(num_agents=1)
        # Set the observation.
        observation = np.concatenate(
            (
                self.sim.get_pipette_position(self.sim.robotIds[0]), 
                self.goal_position
            ), axis=0
        ).astype(np.float32)

        # Reset the number of steps
        self.steps = 0
        self.previous_distance = None

        info = {}

        return observation, info

    def step(self, action):
        # Set the actions
        action = np.append(np.array(action, dtype=np.float32), 0)
        
        # Call the step function
        observation = self.sim.run([action])
        pipette_position = self.sim.get_pipette_position(self.sim.robotIds[0])
        
        # Process observation
        observation = np.array(pipette_position, dtype=np.float32)
        
        # Calculate distance to the goal
        distance = np.linalg.norm(np.array(pipette_position) - np.array(self.goal_position))
        
        # Maximum possible distance in the workspace
        max_distance = np.linalg.norm([
            self.x_max - self.x_min,
            self.y_max - self.y_min,
            self.z_max - self.z_min
        ])
        
        # Reward function
        reward = 0
        
        # Proportional distance reward (closer is better), normalized by max distance
        normalized_distance = distance / max_distance
        reward += 10.0 * (1.0 - normalized_distance)  # Higher reward as normalized distance decreases
        
        # Penalty for taking a step
        reward -= 0.05  # Small penalty per step
        
        # Efficiency reward for significant progress
        if self.previous_distance is not None:
            progress = self.previous_distance - distance
            if progress > 0:  # Only reward progress
                reward += progress * 10  # Scale based on progress magnitude
        
        # Save current distance as previous for next step
        self.previous_distance = distance
        
        # Bonus for reaching the goal
        if distance <= 0.001:  # Within 1 mm
            reward += 50.0  # Large reward for completing the task
            terminated = True
        else:
            terminated = False
        
        # Truncate if max steps exceeded
        if self.steps >= self.max_steps:
            truncated = True
        else:
            truncated = False
        
        # Update observation with current and goal positions
        observation = np.concatenate((pipette_position, self.goal_position), axis=0).astype(np.float32)
        info = {}
        
        # Increment step count
        self.steps += 1
        
        return observation, reward, terminated, truncated, info

    def get_plate_image(self):
        return self.sim.get_plate_image()

    def render(self, mode='human'):
        pass

    def close(self):
        self.sim.close()
