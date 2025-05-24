import os
import numpy as np
from scipy.ndimage import uniform_filter, median_filter, gaussian_filter
from filter_parts import StructureOrientedFiltering, StructureOrientedMeanFiltering, StructureOrientedMedianFiltering


# 定义不同的滤波方法
def apply_filters(array):
    filters = {
        # Mean filter
        "uniform_filter": uniform_filter(array, size=(5, 5, 5)),

        # Median filter
        "median_filter": median_filter(array, size=(5, 5, 5)),

        # Gaussian filter
        "gaussian_filter": gaussian_filter(array, sigma=1.5),

        # Structure-oriented mean filter
        "structure_oriented_mean_filtering": StructureOrientedMeanFiltering(array, r1=2, r2=2, eps=0.01, order=2),

        # Structure-oriented median filter
        "structure_oriented_medain_filtering": StructureOrientedMedianFiltering(array, r1=2, r2=2, eps=0.01, order=2),

        # Structure-oriented filter
        "anisotropic_diffusion_filter": StructureOrientedFiltering(array, niter=10, kappa=20, gamma=0.1,
                                                                   step=(2., 2., 1.), sigma=1.0, option=2),

    }
    return filters


# Recursively traverse the folder and apply filters
def process_images(input_dir, output_dir):
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.endswith(".npy"):
                # Read the npy array
                npy_path = os.path.join(root, file)
                npy_array = np.load(npy_path)

                # Apply filters
                filtered_arrays = apply_filters(npy_array)

                # Save the filtered results as npy files
                for filter_name, filtered_array in filtered_arrays.items():
                    # Build the output folder path
                    relative_path = os.path.relpath(root, input_dir)
                    output_folder = os.path.join(output_dir, filter_name, relative_path)
                    os.makedirs(output_folder, exist_ok=True)

                    # Save the npy file
                    output_npy_path = os.path.join(output_folder, file)
                    np.save(output_npy_path, filtered_array)


# Define input and output directories
input_dir = "seismic"
output_dir = "filter"

# Process the images
process_images(input_dir, output_dir)
