import os
import cv2
import numpy as np


def npy_restoration(input_dir, output_root_dir):
    # Traverse through the input folder
    for root, dirs, files in os.walk(input_dir):
        # For each file, check if it's a png file
        for file in files:
            if file.endswith('.png'):
                # Get the full file path
                image_path = os.path.join(root, file)

                # Read the png file
                image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                image = (image > 0).astype(np.float32)  # Convert 255 to 1
                image = np.transpose(image)  # Swap two dimensions

                # Generate the output path maintaining the same directory structure
                relative_path = os.path.relpath(root, input_dir)  # Relative path
                output_sub_dir = os.path.join(output_root_dir, relative_path)  # Output subfolder path
                os.makedirs(output_sub_dir, exist_ok=True)  # Create output subfolder if it doesn't exist
                output_dir = os.path.join(output_sub_dir, file.replace('.png', '.npy'))  # Output file path

                np.save(output_dir, image)  # Save as npy file


# Define input and output directories
input_dir = 'edge_png'  # Folder containing the npy slices
output_dir = 'edge_npy'  # Folder for saving the images
npy_restoration(input_dir, output_dir)

print(f"All edge images have been restored as npy arrays and saved to folder: {output_dir}")
