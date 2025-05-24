import os
import numpy as np
from matplotlib import pyplot as plt


# Save validation and results
def save_valid_result(segs, inputs, gts, args):
    result_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'results', args.exp,
                               args.mode)  # Path to save validation results
    path_original_numpy = os.path.join(result_path, 'numpy', 'original')  # Path to save original npy arrays
    path_label_numpy = os.path.join(result_path, 'numpy', 'label')  # Path to save label npy arrays
    path_segmentation_numpy = os.path.join(result_path, 'numpy', 'segmentation')  # Path to save predicted npy arrays
    path_segmentation_numpy_opti = os.path.join(result_path, 'numpy', 'segmentation_opti')  # Path to save optimized predicted npy arrays
    path_original_picture = os.path.join(result_path, 'picture', 'original')  # Path to save original image results
    path_label_picture = os.path.join(result_path, 'picture', 'label')  # Path to save label image results
    path_segmentation_picture = os.path.join(result_path, 'picture', 'segmentation')  # Path to save predicted image results
    path_segmentation_picture_opti = os.path.join(result_path, 'picture', 'segmentation_opti')  # Path to save optimized predicted image results

    # Create directories if they don't exist
    os.makedirs(path_original_numpy, exist_ok=True)
    os.makedirs(path_label_numpy, exist_ok=True)
    os.makedirs(path_segmentation_numpy, exist_ok=True)
    os.makedirs(path_segmentation_numpy_opti, exist_ok=True)
    os.makedirs(path_original_picture, exist_ok=True)
    os.makedirs(path_label_picture, exist_ok=True)
    os.makedirs(path_segmentation_picture, exist_ok=True)
    os.makedirs(path_segmentation_picture_opti, exist_ok=True)

    # Iterate through batches
    for i in range(len(inputs)):
        seg_original = segs[i]
        seg = seg_original.argmax(axis=1)  # Model prediction result (index with highest probability along axis)
        img = inputs[i]  # Model input
        gt = gts[i]  # Ground truth label

        # Remove batch and channel dimensions
        seg = np.squeeze(seg)
        img = np.squeeze(img)
        gt = np.squeeze(gt)

        # Save npy results
        np.save(os.path.join(path_original_numpy, str(i + 1) + '.npy'), img)  # Original result
        np.save(os.path.join(path_label_numpy, str(i + 1) + '.npy'), gt)  # Label result
        np.save(os.path.join(path_segmentation_numpy, str(i + 1) + '.npy'), seg)  # Prediction result

        # Save images using matplotlib colormap
        plt.imsave(
            os.path.join(path_original_picture, str(i + 1) + '.png'),
            img.T,
            cmap='seismic',
            format='png'
        )
        plt.imsave(
            os.path.join(path_label_picture, str(i + 1) + '.png'),
            gt.T,
            cmap='binary',
            format='png'
        )
        plt.imsave(
            os.path.join(path_segmentation_picture, str(i + 1) + '.png'),
            seg.T,
            cmap='binary',
            format='png'
        )

        # The stratigraphy is a line, and for each layer, only one result exists vertically, so take the highest probability
        seg_value = seg_original.max(axis=1)
        seg_value = np.squeeze(seg_value)

        # Iterate through each row
        for j in range(seg.shape[0]):
            # Iterate through non-zero classes (starting from 1)
            for cls in range(1, np.max(seg) + 1):
                # Find indices of the current class in the row
                indices = np.where(seg[j] == cls)[0]

                # If there are more than one element for this class
                if len(indices) > 1:
                    # Find the index with the highest probability
                    max_prob_idx = np.argmax(seg_value[j, indices])
                    # Set other elements of this class to 0
                    for k in indices:
                        if k != indices[max_prob_idx]:
                            seg[j, k] = 0

        # Save optimized npy results
        np.save(os.path.join(path_segmentation_numpy_opti, str(i + 1) + '.npy'), seg)  # Optimized prediction result

        # Save optimized image results using matplotlib colormap
        plt.imsave(
            os.path.join(path_segmentation_picture_opti, str(i + 1) + '.png'),
            seg.T,
            cmap='binary',
            format='png'
        )


# Save test results information
def save_test_result(segs, segs_opti, inputs, gts, args):
    result_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'results', args.exp,
                               args.mode)  # Path to save test results
    path_original_numpy = os.path.join(result_path, 'numpy', 'original')  # Path to save original npy arrays
    path_label_numpy = os.path.join(result_path, 'numpy', 'label')  # Path to save label npy arrays
    path_segmentation_numpy = os.path.join(result_path, 'numpy', 'segmentation')  # Path to save predicted npy arrays
    path_segmentation_numpy_opti = os.path.join(result_path, 'numpy', 'segmentation_opti')  # Path to save optimized predicted npy arrays
    path_original_picture = os.path.join(result_path, 'picture', 'original')  # Path to save original image results
    path_label_picture = os.path.join(result_path, 'picture', 'label')  # Path to save label image results
    path_segmentation_picture = os.path.join(result_path, 'picture', 'segmentation')  # Path to save predicted image results
    path_segmentation_picture_opti = os.path.join(result_path, 'picture', 'segmentation_opti')  # Path to save optimized predicted image results

    # Create directories if they don't exist
    os.makedirs(path_original_numpy, exist_ok=True)
    os.makedirs(path_label_numpy, exist_ok=True)
    os.makedirs(path_segmentation_numpy, exist_ok=True)
    os.makedirs(path_segmentation_numpy_opti, exist_ok=True)
    os.makedirs(path_original_picture, exist_ok=True)
    os.makedirs(path_label_picture, exist_ok=True)
    os.makedirs(path_segmentation_picture, exist_ok=True)
    os.makedirs(path_segmentation_picture_opti, exist_ok=True)

    for i in range(len(inputs)):
        # Process each image
        seg = segs[i]
        seg_opti = segs_opti[i]
        img = inputs[i]
        gt = gts[i]

        # Remove batch and channel dimensions
        seg = np.squeeze(seg)
        seg_opti = np.squeeze(seg_opti)
        img = np.squeeze(img)
        gt = np.squeeze(gt)

        # Save npy results
        np.save(os.path.join(path_original_numpy, str(i + 1) + '.npy'), img)  # Original result
        np.save(os.path.join(path_label_numpy, str(i + 1) + '.npy'), gt)  # Label result
        np.save(os.path.join(path_segmentation_numpy, str(i + 1) + '.npy'), seg)  # Prediction result
        np.save(os.path.join(path_segmentation_numpy_opti, str(i + 1) + '.npy'), seg_opti)  # Optimized prediction result

        # Save images using matplotlib colormap
        plt.imsave(
            os.path.join(path_original_picture, str(i + 1) + '.png'),
            img.T,
            cmap='seismic',
            format='png'
        )
        plt.imsave(
            os.path.join(path_label_picture, str(i + 1) + '.png'),
            gt.T,
            cmap='binary',
            format='png'
        )
        plt.imsave(
            os.path.join(path_segmentation_picture, str(i + 1) + '.png'),
            seg.T,
            cmap='binary',
            format='png'
        )
        plt.imsave(
            os.path.join(path_segmentation_picture_opti, str(i + 1) + '.png'),
            seg.T,
            cmap='binary',
            format='png'
        )
