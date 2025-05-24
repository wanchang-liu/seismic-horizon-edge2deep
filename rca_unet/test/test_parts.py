import torch
import numpy as np
from utils.standardization import torch_regularization


# Sliding window test function
def sliding_window_test(input_data, block_size, overlap, model, args):
    _, _, height, width = input_data.shape  # Size of the prediction data
    stride = int(block_size * (1 - overlap))  # Sliding step size

    # Initialize output and weight map
    output = torch.zeros(args.in_channels, args.out_channels, height, width).to(args.device)
    weight_map = torch.zeros(args.in_channels, args.out_channels, height, width).to(args.device)

    for x in range(0, height, stride):
        # Get the current block
        x_end = min(x + block_size, height)
        x_start = x_end - block_size
        block = input_data[:, :, x_start:x_end, :]

        # Regularization
        block_normal = torch_regularization(block)

        # Model prediction
        block_prediction = model(block_normal)

        # Softmax activation
        block_prediction = torch.softmax(block_prediction, dim=1)

        # Update output and weight map
        output[:, :, x_start:x_end, :] += block_prediction
        weight_map[:, :, x_start:x_end, :] += 1

    # Calculate final output
    output /= weight_map

    # Prediction as probability values for the layers
    output_npy = output.detach().cpu().numpy()
    output_prob = output_npy[:, 1, :, :]
    output_prob = np.squeeze(output_prob)

    # Binary classification result
    output_binary = output_npy.argmax(axis=1)  # Returns the index of the maximum value along the specified axis as the prediction result
    output_binary = np.squeeze(output_binary)  # Remove batch and channel dimensions

    # Copy prediction result for optimization
    output_opti = np.copy(output_binary)

    # Optimize results row by row
    for i in range(height):
        for j in range(args.out_channels):
            # Find all indices in the current row belonging to class 1
            indices = np.where(output_binary[i] == j)[0]
            # If there are more than one element of this class
            if len(indices) > 1:
                # Find the index of the element with the highest probability
                max_prob_idx = np.argmax(output_prob[i, indices])
                # Set other elements of this class to 0
                for k in indices:
                    if k != indices[max_prob_idx]:
                        output_opti[i, k] = 0

    return output_prob, output_binary, output_opti
