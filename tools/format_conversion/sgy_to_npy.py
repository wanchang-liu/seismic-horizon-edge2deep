import os
import segyio
import numpy as np


# Convert .sgy file to .npy file
def sgy_to_npy(sgypath, npypath):
    # Open the .sgy file in read-only mode
    with segyio.open(sgypath, 'r') as segyfile:
        # Use segyio tools to convert the .sgy file to a data cube
        data = segyio.tools.cube(segyfile)

    # Save the converted data as a .npy file
    np.save(npypath, data)


# Convert all .sgy files in a directory to .npy files
def save_all_sgy_as_npy(sgy_dir, npy_dir):
    # Create the output directory if it doesn't exist
    os.makedirs(npy_dir, exist_ok=True)

    # Traverse all files in the specified directory
    for filename in os.listdir(sgy_dir):
        if filename.endswith('.sgy'):
            sgy_path = os.path.join(sgy_dir, filename)  # .sgy file path
            npy_filename = os.path.splitext(filename)[0] + '.npy'  # .npy file name
            npy_path = os.path.join(npy_dir, npy_filename)  # .npy file path
            sgy_to_npy(sgy_path, npy_path)
            print(f"File {filename} has been successfully converted to {npy_filename}")


sgy_path = 'sgy'
npy_path = 'npy'
save_all_sgy_as_npy(sgy_path, npy_path)
print("All .sgy files have been successfully converted to .npy files")
