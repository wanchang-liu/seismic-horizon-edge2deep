import cv2
import os
from edge_detection_parts import canny, otsu_canny


def process_images_in_folder(input_folder, output_folder):
    # Traverse the input folder
    for root, dirs, files in os.walk(input_folder):
        for file in files:
            # Only process image files
            if file.endswith('.png'):
                image_path = os.path.join(root, file)

                # Read the image in grayscale mode
                image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

                # Get the relative path of the image file (remove outer input path and inner file name)
                relative_path = os.path.relpath(image_path, input_folder)
                folder_path = os.path.dirname(relative_path)

                # Create an output folder structure that mirrors the input folder structure
                output_dir_base = os.path.join(output_folder, folder_path)

                # Process the image using different edge detection algorithms
                results = {
                    'CannyHigh': canny(image, 75, 150),
                    'CannyLow': canny(image, 25, 50),
                    'Otsu_Canny': otsu_canny(image, lowrate=0.5)
                }

                # Create a folder for each edge detection algorithm and save the results
                for algorithm, result in results.items():
                    output_dir = os.path.join(output_dir_base, algorithm)
                    os.makedirs(output_dir, exist_ok=True)
                    output_path = os.path.join(output_dir, file)
                    cv2.imwrite(output_path, result)


input_folder = 'sections_gray'
output_folder = 'edge'
process_images_in_folder(input_folder, output_folder)
