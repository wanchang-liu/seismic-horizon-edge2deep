import os
import cv2


def image_flipped(input_dir, output_root_dir):
    # Traverse the input directory
    for root, dirs, files in os.walk(input_dir):
        # For each file, check if it is a PNG file
        for file in files:
            if file.endswith('.png'):
                # Get the full file path
                image_path = os.path.join(root, file)

                # Read the PNG file
                image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

                # Generate the output path with the same directory structure as the input
                relative_path = os.path.relpath(root, input_dir)  # Relative path
                output_sub_dir = os.path.join(output_root_dir, relative_path)  # Output subdirectory path
                os.makedirs(output_sub_dir, exist_ok=True)  # Create the output subdirectory if it doesn't exist
                output_dir = os.path.join(output_sub_dir, file)  # Output file path

                cv2.imwrite(output_dir, cv2.flip(image, 1))  # Flip the image horizontally


# Define input and output directories
input_dir = 'edge_invert'  # Folder containing the npy profiles
output_dir = 'edge_flipped'  # Folder to save the flipped images
image_flipped(input_dir, output_dir)

print(f"All edge images have been flipped horizontally and saved to folder: {output_dir}")
