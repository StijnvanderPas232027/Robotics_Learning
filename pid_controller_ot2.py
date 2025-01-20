import time
import numpy as np
from ot2_env_wrapper import OT2Env

# PID Controller Class
class PIDController:
    def __init__(self, kp, ki, kd, setpoint):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.setpoint = setpoint
        self.integral = 0
        self.previous_error = 0

    def compute(self, current_position, dt):
        error = self.setpoint - current_position
        self.integral += error * dt
        derivative = (error - self.previous_error) / dt if dt > 0 else 0
        output = self.kp * error + self.ki * self.integral + self.kd * derivative

        # Anti-windup: Dynamically limit the integral term
        max_integral = 0.5  # Further tightened for precision
        self.integral = max(min(self.integral, max_integral), -max_integral)

        self.previous_error = error
        return output

# Environment Wrapper with PID Integration
class OT2EnvWithPID(OT2Env):
    def __init__(self, render=False, max_steps=500):
        super(OT2EnvWithPID, self).__init__(render, max_steps)

        # Initialize PID controllers for X, Y, Z axes
        self.pid_x = PIDController(kp=3.0, ki=0.2, kd=0.2, setpoint=0.0)
        self.pid_y = PIDController(kp=3.0, ki=0.2, kd=0.2, setpoint=0.0)
        self.pid_z = PIDController(kp=3.5, ki=0.25, kd=0.25, setpoint=0.0)

        # Set a random goal position
        self.goal_position = np.array([
            np.random.uniform(-0.187, 0.253),
            np.random.uniform(-0.1705, 0.2195),
            np.random.uniform(0.1195, 0.2895)
        ], dtype=np.float32)

    def step(self, action):
        # Update PID setpoints with goal position
        self.pid_x.setpoint = self.goal_position[0]
        self.pid_y.setpoint = self.goal_position[1]
        self.pid_z.setpoint = self.goal_position[2]

        # Get current pipette position
        robot_id_key = f'robotId_{self.sim.robotIds[0]}'
        pipette_position = self.sim.get_states()[robot_id_key]['pipette_position']

        # Compute corrections using PID controllers
        dt = 0.005  # Further reduced time step for higher precision
        distance = np.linalg.norm(pipette_position - self.goal_position)

        # Dynamic gain adjustment based on distance
        scaling_factor = max(1.0, min(10.0, distance * 10))  # Scale between 1x and 10x
        self.pid_x.kp = 3.0 * scaling_factor
        self.pid_y.kp = 3.0 * scaling_factor
        self.pid_z.kp = 3.5 * scaling_factor

        correction_x = self.pid_x.compute(pipette_position[0], dt)
        correction_y = self.pid_y.compute(pipette_position[1], dt)
        correction_z = self.pid_z.compute(pipette_position[2], dt)

        # Combine corrections into action
        pid_action = [correction_x, correction_y, correction_z, action[-1]]  # Include drop action

        # Execute the action in the simulation
        observation, reward, terminated, truncated, info = super().step(pid_action)

        # Modify reward based on PID performance
        distance = np.linalg.norm(pipette_position - self.goal_position)
        reward = -distance  # Reward is negative distance to goal

        # Print current distance to goal in millimeters
        distance_mm = distance * 1000  # Convert meters to millimeters
        print(f"Step {self.steps}: Distance to Goal: {distance_mm:.3f} mm")

        # Termination condition: 1 mm threshold for accuracy
        termination_threshold = 0.001
        terminated = distance < termination_threshold

        # Early exit for non-convergence: stop if not reducing error
        if self.steps > 300 and distance_mm > 10:  # Error plateau
            terminated = True
            print("Terminating early due to non-convergence.")

        return observation, reward, terminated, truncated, info

# Simulation Function
if __name__ == "__main__":
    # Initialize environment with PID integration
    env = OT2EnvWithPID(render=False, max_steps=500)

    # Reset environment
    obs, _ = env.reset()

    # Simulate steps
    for step in range(500):
        # Example action: PID corrections
        action = [0, 0, 0, 0]  
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"Step {step}: Reward = {reward:.4f}, Terminated = {terminated}")

        if terminated or truncated:
            break

    print("\nSimulation complete.")
    env.close()
