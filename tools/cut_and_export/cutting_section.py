import os
import numpy as np


def save_sections(input_dir, output_root_dir):
    for file_name in os.listdir(input_dir):
        if file_name.endswith('.npy'):
            input_file = os.path.join(input_dir, file_name)
            seismic_data = np.load(input_file)

            # Get the shape of the data
            n_inline, n_crossline, n_time = seismic_data.shape

            # Create output subfolders corresponding to npy file names
            output_dir = os.path.join(output_root_dir, os.path.splitext(file_name)[0])
            os.makedirs(output_dir, exist_ok=True)

            # Save all inline sections
            for i in range(n_inline):
                inline_section = seismic_data[i, :, :]
                np.save(os.path.join(output_dir, f'inline_{i + 1}.npy'), inline_section)

            # Save all crossline sections
            for j in range(n_crossline):
                crossline_section = seismic_data[:, j, :]
                np.save(os.path.join(output_dir, f'crossline_{j + 1}.npy'), crossline_section)

            print(f"All inline and crossline sections of file {file_name} have been saved to folder: {output_dir}")


input_dir = 'npy'  # 3D npy cube file directory
output_root_dir = 'sections'  # Root directory of 2D npy profile files
save_sections(input_dir, output_root_dir)
print(f"All sections have been exported")
