import os
import torch
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


# Seismic dataset construction
class SeismicDataset(Dataset):
    # Initialization function, accepts dataset path 'path', mode ('train' by default), and transform (default is None, i.e., no data transformation)
    def __init__(self, path, transform=None):
        # Parameter initialization
        self.path = path
        self.transform = transform
        self.image_list, self.label_list = self.load_data()

    # Fetch the data item at index 'index'
    def __getitem__(self, index):
        # Load image data
        image = np.load(self.image_list[index])
        label = np.load(self.label_list[index])

        # Reshape the image to (channels, height, width) format
        image = image.reshape((1, image.shape[0], image.shape[1]))

        x = torch.from_numpy(image)  # Convert image to PyTorch tensor
        y = torch.from_numpy(label)  # Convert label to PyTorch tensor

        data = {'x': x.float(), 'y': y.float()}  # Create data dictionary containing image and label

        return data

    # Return the length of the dataset
    def __len__(self):
        return len(self.image_list)

    # Function to load data
    def load_data(self):
        img_list = []  # List to store image file paths
        label_list = []  # List to store label file paths
        img_path = os.path.join(self.path, 'x')  # Image path
        label_path = os.path.join(self.path, 'y')  # Label path

        # Iterate through all filenames in the image path
        for item in os.listdir(img_path):
            img_list.append(os.path.join(img_path, item))  # Add full image file path to the list
            label_list.append(os.path.join(label_path,
                                           item))  # Add corresponding label file path to the list (since x and y filenames are the same, they are loaded in one step)

        return img_list, label_list


# Load seismic data
def load_data(args):
    # Training mode
    if args.mode == 'train':
        # Load training data
        train_dataset = SeismicDataset(args.train_path, transform=None)
        # train_dataset: This is the dataset to load.
        # batch_size: Specifies the number of samples per batch.
        # shuffle: If True, the data will be shuffled during each epoch.
        # num_workers: Specifies the number of subprocesses to use for data loading.
        # drop_last: If True, drops the last incomplete batch if the dataset size is not divisible by the batch size.
        train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers,
                                      drop_last=True)

        print('--- Training data loader created ---')
        print(len(train_dataset), ': Training dataset created')
        print(len(train_dataloader), ': Training data loader created')

        # Load validation data
        valid_dataset = SeismicDataset(args.valid_path, transform=None)
        valid_dataloader = DataLoader(valid_dataset, batch_size=args.batch_size_not_train, shuffle=True,
                                      num_workers=args.workers, drop_last=True)

        print('--- Validation data loader created ---')
        print(len(valid_dataset), ': Validation dataset created')
        print(len(valid_dataloader), ': Validation data loader created')

        return train_dataloader, valid_dataloader

    # Validation mode
    elif args.mode == 'valid':
        dataset = SeismicDataset(args.valid_path, transform=None)
        dataloader = DataLoader(dataset, batch_size=args.batch_size_not_train, shuffle=True, num_workers=args.workers,
                                drop_last=True)

        print('--- Validation data loader created ---')
        print(len(dataset), ': Validation dataset created')
        print(len(dataloader), ': Validation data loader created')

        return dataloader

    # Test mode
    elif args.mode == 'test':
        dataset = SeismicDataset(args.test_path, transform=None)
        dataloader = DataLoader(dataset, batch_size=args.batch_size_not_train, shuffle=False, num_workers=args.workers,
                                drop_last=True)

        print('--- Test data loader created ---')
        print(len(dataset), ': Test dataset created')
        print(len(dataloader), ': Test data loader created')

        return dataloader
