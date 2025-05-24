# Seismic Horizon Tracking
This repository contains code for seismic horizon tracking, which is divided into two main parts:

Overall Horizon Extraction: Implemented using an improved Canny edge detection algorithm.

Target Horizon Tracking: Implemented using an improved UNet architecture.

The code is fully open-source, and all functionality is commented in the code.py file. For detailed explanations of the specific functions, please refer to the comments in the code itself.

# Dataset
The original seismic dataset used in this study is publicly available from the Queensland Government at the following link:

Queensland Seismic Data

Due to the large size of the dataset, you will need to download it from the provided link.

Note: The horizon labels used for training the model were manually annotated and are not open-source. You can either manually annotate your own labels, ask an expert to do so, or use publicly available seismic datasets with high-quality labels. One such resource is the following:

DGBES Seismic Datasets

# Model Training
The model training involves using the seismic data, where the labels (horizon annotations) are manually generated. Please refer to the relevant sections in the code for setup and instructions on model training.

# License
This repository is open-source under the MIT License. Please refer to the LICENSE file for more details.

Feel free to modify any part as needed!
