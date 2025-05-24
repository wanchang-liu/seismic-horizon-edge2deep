import cv2
import os
from edge_detection_parts import roberts, prewitt, sobel, scharr, laplacian


def process_images_in_folder(input_folder, output_folder):
    # Traverse the input folder
    for root, dirs, files in os.walk(input_folder):
        for file in files:
            # Only process image files
            if file.endswith('.png'):
                image_path = os.path.join(root, file)

                # Read the image in grayscale mode
                image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

                # Get the relative path of the image file (remove the outer input path and inner file name)
                relative_path = os.path.relpath(image_path, input_folder)
                folder_path = os.path.dirname(relative_path)

                # Create an output folder with the same inner directory structure as the input folder
                output_dir_base = os.path.join(output_folder, folder_path)

                # Gaussian smoothing to remove noise
                blur = cv2.GaussianBlur(image, (3, 3), 0)

                # Use different edge detection operators to enhance image edges
                results = {
                    'Roberts': cv2.addWeighted(image, 1.0, roberts(blur), -0.5, 0),
                    'Prewitt': cv2.addWeighted(image, 1.0, prewitt(blur), -0.5, 0),
                    'Sobel': cv2.addWeighted(image, 1.0, sobel(blur), -0.5, 0),
                    'Scharr': cv2.addWeighted(image, 1.0, scharr(blur), -0.5, 0),
                    'Laplacian': cv2.addWeighted(image, 1.0, laplacian(blur), -0.5, 0),
                }

                # Create a folder for each edge detection algorithm and save the results
                for algorithm, result in results.items():
                    output_dir = os.path.join(output_dir_base, algorithm)
                    os.makedirs(output_dir, exist_ok=True)
                    output_path = os.path.join(output_dir, file)
                    cv2.imwrite(output_path, result)


input_folder = 'sections_gray'
output_folder = 'gray_enhancement'
process_images_in_folder(input_folder, output_folder)
