import os
import numpy as np
import matplotlib.pyplot as plt


def export_sections_to_images(input_dir, output_dir):
    # Traverse the input folder
    for root, dirs, files in os.walk(input_dir):
        # For each file, check if it is a npy file
        for file in files:
            if file.endswith('.npy'):
                # Get the full file path
                npy_path = os.path.join(root, file)

                # Read npy file
                section_data = np.load(npy_path)

                # Calculate the absolute maximum value of an array
                max_val = np.max(np.abs(section_data))

                # Generate an output path with the same directory structure as the input
                relative_path = os.path.relpath(root, input_dir)  # Relative paths
                output_sub_dir = os.path.join(output_dir, relative_path)  # Output subfolder path

                # If the output subfolder does not exist, create it
                os.makedirs(output_sub_dir, exist_ok=True)

                # Full path to the saved image file (use the same file name, but with a png extension)
                output_image_path = os.path.join(output_sub_dir, file.replace('.npy', '.png'))

                # Save directly to image using matplotlib's gray colormap
                plt.imsave(
                    output_image_path,
                    section_data.T,
                    cmap='gray',
                    vmin=-max_val,
                    vmax=max_val,
                    format='png'
                )


# Define input and output folders
input_dir = 'sections'  # Folder where npy profiles are stored
output_dir = 'sections_gray'  # The folder where the pictures are stored
export_sections_to_images(input_dir, output_dir)
print(f"All sections have been exported as gray png images and saved to the folder: {output_dir}")
