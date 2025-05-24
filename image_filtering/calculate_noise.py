import os
import numpy as np

# Define the paths for the original and filtered data folders
original_dir = "seismic"
filter_dir = "filter"

# Output folder
output_file = "noise"

# Traverse through the input folder
for root, dirs, files in os.walk(filter_dir):
    # For each file, check if it is a .npy file
    for file in files:
        if file.endswith('.npy'):
            # Get the full path of the filtered data
            filter_path = os.path.join(root, file)

            # Load the filtered data
            filter_data = np.load(filter_path)

            # Original data path
            original_path = os.path.join(original_dir, file)

            # Load the original data
            original_data = np.load(original_path)

            # Calculate the noise data (original - filtered)
            noise_data = original_data - filter_data

            # Generate output path maintaining the same directory structure
            relative_path = os.path.relpath(root, filter_dir)
            output_sub_dir = os.path.join(output_file, relative_path)

            # Create the subdirectory if it doesn't exist
            os.makedirs(output_sub_dir, exist_ok=True)

            # Define the path for saving the noise data
            noise_path = os.path.join(output_sub_dir, file)

            # Save the noise data as a .npy file
            np.save(noise_path, noise_data)
