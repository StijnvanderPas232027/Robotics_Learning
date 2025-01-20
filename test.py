import cv2
import matplotlib.pyplot as plt

from stable_baselines3 import PPO
import numpy as np
from ot2_env_wrapper_task13 import OT2Env


from stable_baselines3 import PPO
import numpy as np

# Initialise the simulation environment
# Initialize the environment
env = OT2Env(render=True, max_steps=1000)

# Reset the environment
observation, info = env.reset()

# Get the plate image (assumes the Simulation class has a get_plate_image method implemented)
plate_image = env.get_plate_image()

# Do something with the plate image
print("Plate image:", plate_image)

# Close the environment when done
env.close()


def process_image(input_path, output_path):
    # Load the image
    img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)

    # Threshold the image to isolate the object of interest
    _, thresh = cv2.threshold(img, 50, 255, cv2.THRESH_BINARY)

    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Get bounding box of the largest contour
    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)

    # Crop the image to the bounding box
    cropped_img = img[y:y+h, x:x+w]

    # Resize to a square format
    final_size = 3072
    resized_img = cv2.resize(cropped_img, (final_size, final_size))

    # Save the processed image
    cv2.imwrite(output_path, resized_img)

    return resized_img

def process_and_display_image(input_image_path, output_image_path):
    # Process the input image
    processed_img = process_image(input_image_path, output_image_path)

    # Display the processed image
    plt.figure(figsize=(6, 6))
    plt.imshow(processed_img, cmap='gray')
    plt.title(f"Processed Image ({processed_img.shape[1]}, {processed_img.shape[0]})")
    plt.axis('off')
    plt.show()

# Example usage
input_image_path = plate_image # Replace with the path to your input image
output_image_path = "path_to_save_processed_image.png"  # Replace with the desired output path

process_and_display_image(input_image_path, output_image_path)
