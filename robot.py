from sim_class import Simulation
import time

# Initialize the simulation
sim = Simulation(num_agents=1)  # Initialize one robot

# Define the velocities for moving to each corner
velocities = [
    [0.0, 0.1, 0.0],  # Move along +Y
    [0.1, 0.0, 0.0],  # Move along +X
    [0.0, 0.0, 0.1],  # Move along +Z
    [-0.1, 0.0, 0.0],  # Move along -X
    [0.0, -0.1, 0.0],  # Move along -Y
    [0.1, 0.0, 0.0],  # Move along +X
    [0.0, 0.0, -0.1],  # Move along -Z
    [-0.1, 0.0, 0.0],  # Move along -X
]

# Main loop to navigate to all corners
try:
    for velocity in velocities:
        print(f"Moving with velocity: {velocity}")

        # Apply the velocity for a fixed duration
        for _ in range(500):  # Adjust range for distance covered
            actions = [[velocity[0], velocity[1], velocity[2], 0]]  # No drop command
            sim.run(actions, num_steps=1)
            time.sleep(0.01)  # Delay for visualization

        # Get and print the robot's position
        current_position = sim.get_robot_position()
        print(f"Reached position: {current_position}")

    print("Finished moving to all corners. Keeping the simulation open.")

    # Keep the simulation open for inspection
    while True:
        sim.run([[0.0, 0.0, 0.0, 0]], num_steps=1)  # Robot stays still
        time.sleep(0.1)  # Avoid busy-waiting

except KeyboardInterrupt:
    print("Exiting simulation on user interrupt...")
