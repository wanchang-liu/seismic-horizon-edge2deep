import segyio
import numpy as np


# Convert .npy file to .segy file
def npy_to_sgy(npyfilepath, sgyfilepath):
    # Load the .npy file
    npyfile = np.load(npyfilepath)

    # Reshape the array into 2D, automatically calculating the number of rows,
    # with the number of columns as the original array's third dimension
    npyfile = npyfile.reshape(-1, npyfile.shape[2])

    # Open the .segy file in read-write mode
    with segyio.open(sgyfilepath, 'r+') as segyfile:
        # Iterate over each trace
        for i in range(len(segyfile.trace)):
            # Assign data from the .npy file to the .segy file's trace
            segyfile.trace[i] = npyfile[i]


npy_file_path = 'UMIGR.npy'
sgy_file_path = 'UMIGR.sgy'
npy_to_sgy(npy_file_path, sgy_file_path)
