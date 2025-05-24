import os
import torch
import numpy as np
from tqdm import tqdm
from utils.create_model import create_model
from predict.predict_parts import sliding_window_prediction


# Prediction function
def pred(args):
    print('------------------')
    print("Starting prediction")

    # Set the computation device (GPU or CPU)
    device = torch.device(args.device)
    print('------------------')
    print('Computation device is:', device)

    # Load the pre-trained model
    print('------------------')
    print("Loading model")
    model = create_model(args)
    model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'train_models', args.exp,
                              args.pretrained_model_name)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    print("Model loaded successfully")
    model.eval()

    # Create prediction file storage paths
    pred_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'results', args.exp, args.mode)
    prob_folder = os.path.join(pred_path, 'prob')
    binary_folder = os.path.join(pred_path, 'binary')
    opti_folder = os.path.join(pred_path, 'opti')
    os.makedirs(prob_folder, exist_ok=True)
    os.makedirs(binary_folder, exist_ok=True)
    os.makedirs(opti_folder, exist_ok=True)

    # Predict all profiles one by one
    print('---')
    print("Starting prediction")

    for filename in tqdm(os.listdir(args.pred_path), desc='[Pred] Pred'):
        input_file_path = os.path.join(args.pred_path, filename)
        prediction_prob, prediction_binary, prediction_opti = sliding_window_prediction(input_file_path, 512,
                                                                                        args.overlap, model,
                                                                                        args)  # Model prediction

        # Save prediction results
        np.save(os.path.join(prob_folder, filename), prediction_prob)
        np.save(os.path.join(binary_folder, filename), prediction_binary)
        np.save(os.path.join(opti_folder, filename), prediction_opti)
