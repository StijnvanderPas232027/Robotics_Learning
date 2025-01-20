import gymnasium as gym
from gymnasium import spaces
import numpy as np
from sim_class import Simulation
import pybullet as p

class OT2Env(gym.Env):
    def __init__(self, render=True, max_steps=1000):
        super(OT2Env, self).__init__()
        self.render = render
        self.max_steps = max_steps

        # Create the simulation environment
        self.sim = Simulation(num_agents=1, render=True)

        # Define action and observation space
        # Action space: controlling pipette position (x, y, z) and drop action
        self.action_space = spaces.Box(
            low=np.array([-1, -1, -1, -1], dtype=np.float32),
            high=np.array([1, 1, 1, 1], dtype=np.float32),
            dtype=np.float32
        )

        # Observation space: pipette position (x, y, z) + goal position (x, y, z)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(6,),
            dtype=np.float32
        )

        # Keep track of the number of steps
        self.steps = 0

    def reset(self, seed=None):
        # Set seed for reproducibility
        if seed is not None:
            np.random.seed(seed)

        # Reset the state of the environment to an initial state
        self.goal_position = np.array([
            np.random.uniform(-0.187, 0.253),
            np.random.uniform(-0.1705, 0.2195),
            np.random.uniform(0.1195, 0.2895)
        ], dtype=np.float32)

        # Reset the simulation
        observation = self.sim.reset(num_agents=1)

        # Debug: Print observation structure to verify keys
        print(f"Observation: {observation}")  # Verify the keys in the observation

        # Extract pipette position and form the full observation
        robot_id_key = f'robotId_{self.sim.robotIds[0]}'  # Use the correct robot ID key
        pipette_position = np.array(
            observation[robot_id_key]['pipette_position'],  # Correct key format
            dtype=np.float32
        )
        full_observation = np.concatenate([pipette_position, self.goal_position], axis=0)

        # Reset step counter
        self.steps = 0

        return full_observation, {}

    def step(self, action):
        # Append 0 for the drop action (assuming it's not used yet)
        action = np.append(action, 0)

        # Execute the action in the simulation
        observation = self.sim.run([action])

        # Extract pipette position
        robot_id_key = f'robotId_{self.sim.robotIds[0]}'
        pipette_position = np.array(
            observation[robot_id_key]['pipette_position'],
            dtype=np.float32
        )

        # Form the full observation
        full_observation = np.concatenate([pipette_position, self.goal_position], axis=0)

        # Calculate the reward (negative distance to goal)
        reward = -np.linalg.norm(pipette_position - self.goal_position)

        # Check termination condition (task completion)
        distance = np.linalg.norm(pipette_position - self.goal_position)
        terminated = distance < 0.001  # Adjust threshold based on your task requirements

        # Check truncation condition (exceeded max steps)
        truncated = self.steps >= self.max_steps

        # Increment step counter
        self.steps += 1

        # Return step information
        info = {}
        return full_observation, reward, terminated, truncated, info
    
    
    def get_plate_image(self):
        return self.sim.get_plate_image()
    
    def render(self, mode='human'):
        # Optional: implement rendering logic if needed
        pass

    def close(self):
        # Close the simulation environment
        self.sim.close()