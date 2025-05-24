import os
import cv2
import numpy as np


def edge_optimization(input_dir, output_root_dir):
    # Traverse through the input folder
    for root, dirs, files in os.walk(input_dir):
        # For each file, check if it's a png file
        for file in files:
            if file.endswith('.png'):
                # Get the full file path
                image_path = os.path.join(root, file)

                # Read the png file
                image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

                # Generate the output path maintaining the same directory structure
                relative_path = os.path.relpath(root, input_dir)  # Relative path
                output_sub_dir = os.path.join(output_root_dir, relative_path)  # Output subfolder path
                os.makedirs(output_sub_dir, exist_ok=True)  # Create the output subfolder if it doesn't exist
                output_dir = os.path.join(output_sub_dir, file)  # Output file path

                # Find connected components
                num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(image, connectivity=8)

                # Create an empty image to store the result
                cleaned_edges = np.zeros_like(image)

                # Iterate through all connected components and remove areas smaller than the threshold
                min_area = 20  # Set the minimum area threshold
                for i in range(1, num_labels):  # Ignore the background
                    if stats[i, cv2.CC_STAT_AREA] >= min_area:
                        cleaned_edges[labels == i] = 255

                cv2.imwrite(output_dir, cleaned_edges)  # Remove isolated points


# Define input and output directories
input_dir = 'edge'  # Folder containing the npy slices
output_dir = 'edge_optimization'  # Folder for saving the images
edge_optimization(input_dir, output_dir)

print(f"All edge images have been processed to remove isolated points and saved to folder: {output_dir}")
