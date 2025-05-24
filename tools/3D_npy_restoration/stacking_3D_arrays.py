import numpy as np
import os
from glob import glob

# Path for the 2D npy arrays
folder_path = "edge_npy"
out_path = "edge_3d"
os.makedirs(out_path, exist_ok=True)  # Create the output folder if it doesn't exist

# Read crossline files and reconstruct the 3D array
crossline_files = sorted(glob(os.path.join(folder_path, "crossline_*.npy")), key=lambda x: int(x.split('_')[-1].split('.')[0]))
crossline_slices = [np.load(file) for file in crossline_files]
crossline_array = np.stack(crossline_slices, axis=1)  # Stack along the crossline dimension

# Read inline files and reconstruct the 3D array
inline_files = sorted(glob(os.path.join(folder_path, "inline_*.npy")), key=lambda x: int(x.split('_')[-1].split('.')[0]))
inline_slices = [np.load(file) for file in inline_files]
inline_array = np.stack(inline_slices, axis=0)  # Stack along the inline dimension

# Compute the pointwise union and intersection of the two 3D arrays
union_array = np.logical_or(crossline_array, inline_array).astype(np.float32)  # Union
intersection_array = np.logical_and(crossline_array, inline_array).astype(np.float32)  # Intersection

# Save the results
np.save(os.path.join(out_path, "crossline.npy"), crossline_array)
np.save(os.path.join(out_path, "inline.npy"), inline_array)
np.save(os.path.join(out_path, "union.npy"), union_array)
np.save(os.path.join(out_path, "intersection.npy"), intersection_array)

print("3D array reconstruction completed!")
