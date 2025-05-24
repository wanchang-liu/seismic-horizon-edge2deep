import numpy as np
from scipy.ndimage import label, generate_binary_structure

# Load the npy array
np_array = np.load("union.npy")

# Create a structuring element (3D neighborhood connectivity)
structure = generate_binary_structure(3, 2)

# Use connected components labeling
labeled_array, num_features = label(np_array, structure)

# Compute the volume of each connected region (i.e., the number of pixels in that region)
sizes = np.bincount(labeled_array.ravel())

# Regions smaller than a certain threshold (isolated 1s) will be marked as 0
threshold = 20  # Threshold is set to 20, meaning regions smaller than 20 pixels will be removed
for i in range(1, num_features + 1):
    if sizes[i] < threshold:
        labeled_array[labeled_array == i] = 0

# Regenerate the 3D array after removing isolated 1s
cleaned_array = (labeled_array > 0).astype(np.float32)

np.save("union_cleaned.npy", cleaned_array)
