import torch
import numpy as np
from utils.standardization import regularization


# Sliding window prediction function
def sliding_window_prediction(input_data, block_size, overlap, model, args):
    input_data = np.load(input_data)  # Load prediction data
    height, width = input_data.shape  # Size of the prediction data
    stride = int(block_size * (1 - overlap))  # Sliding step size

    # Initialize output and weight map
    output = np.zeros((1, 2, height, width), dtype=np.float32)
    weight_map = np.zeros((1, 2, height, width), dtype=int)

    for x in range(0, height, stride):
        # Get the current block
        x_end = min(x + block_size, height)
        x_start = x_end - block_size
        block = input_data[x_start:x_end, :]

        # Reshape to [batch, channels, width, height]
        block = block.reshape(1, 1, block_size, block_size)

        # Regularization
        block_normal = regularization(block)

        # Convert to tensor
        input_block = torch.from_numpy(block_normal).to(args.device).float()

        # Model prediction
        block_prediction = model(input_block)

        # Apply softmax activation
        block_prediction = torch.softmax(block_prediction, dim=1)

        # Move prediction result to CPU
        block_prediction = block_prediction.detach().cpu().numpy()

        # Update output and weight map
        output[:, :, x_start:x_end, :] += block_prediction
        weight_map[:, :, x_start:x_end, :] += 1

    # Calculate final output
    output /= weight_map

    # Prediction as probability values for the layers
    output_prob = output[:, 1, :, :]
    output_prob = np.squeeze(output_prob)

    # Binary classification result
    output_binary = output.argmax(axis=1)  # Returns the index of the maximum value along the specified axis as the prediction result
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
