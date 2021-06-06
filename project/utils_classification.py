import os
import gzip
import tarfile

import pickle
import pandas as pd

import numpy as np
import matplotlib.pyplot as plt

import cv2
from skimage import img_as_ubyte
from skimage.util import invert

# Torch
import torch
from torch import nn
import torch.nn.functional as F

# Model
from torchvision.models import resnet18

# Data Loading
from torchvision import datasets
from torch.utils import data

# Data augmentation
from torch import tensor
import torchvision.transforms as transforms


# ------------------------- Load Data -------------------------

def extract_data(filename, image_shape, image_number):
    """Extract MNIST images from the corresponding compressed file

    Parameters
    ----------
    filename : str
        Filepath of file containing MNIST data
    image_shape : (int, int)
        Tuple describing the shape of the image
    image_number : int
        Number of images in the dataset

    Returns
    -------
    np.ndarray
        Numpy array containing the MNIST images in the file
    """
    with gzip.open(filename) as bytestream:
        bytestream.read(16)
        buf = bytestream.read(np.prod(image_shape) * image_number)
        data = np.frombuffer(buf, dtype=np.uint8)
        data = data.reshape(image_number, image_shape[0], image_shape[1])
    return data


def extract_labels(filename, image_number):
    """Extract MNIST labels from the corresponding compressed file

    Parameters
    ----------
    filename : str
        Filepath of file containing MNIST labels
    image_number : int
        Number of images in the dataset

    Returns
    -------
    np.ndarray
        Numpy array containing the MNIST labels in the file
    """
    with gzip.open(filename) as bytestream:
        bytestream.read(8)
        buf = bytestream.read(1 * image_number)
        labels = np.frombuffer(buf, dtype=np.uint8).astype(np.int64)
    return labels


def augment_data(data, transform):
    """Augment the given images with the transform passed as argument

    Parameters
    ----------
    data : np.ndarray
        Images to augment
    transform : torchvision.transforms.Compose
        A composition of transformations to apply on images

    Returns
    -------
    np.ndarray
        Numpy array containing augmented images
    """
    augmented_imgs = []
    image_shape = data[0].shape
    for img in data:
        augmented = img_as_ubyte(transform(img)).reshape(image_shape)
        augmented_imgs.append(augmented)
        
    return np.array(augmented_imgs)


class MNISTDataset(data.Dataset):
    """MNIST Dataset class"""
    def __init__(self, imgs, labels, transform):
        self.imgs = imgs
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        data = self.imgs[idx]
        if self.transform:
            data = self.transform(data)
        return (data, self.labels[idx])

# ------------------------- Load Task 0 -------------------------

def load_segmentation_task0(filepath):
    segmented = pd.read_pickle(filepath)
    values = segmented[0].values
    imgs = [v[-1] for v in values]
    ranks = [v[0][0] for v in values]
    suits = [v[0][1] for v in values]

    df_task0 = pd.DataFrame({"rank": ranks, "suit": suits, "image": imgs})
    df_numbers_only = df_task0[~df_task0['rank'].isin(['Q', 'J', 'K'])].reset_index(drop=True)
    df_not_numbers = df_task0[df_task0['rank'].isin(['Q', 'J', 'K'])].reset_index(drop=True)
    
    return df_task0, df_numbers_only, df_not_numbers
  
def preprocess_segmented_task0(img):
    inverted = img_as_ubyte(invert(img))
    _, binary = cv2.threshold(inverted, 100, 120, cv2.THRESH_BINARY)
    blurred = cv2.GaussianBlur(binary, (3,3), 0)

    return blurred

# ---------------------------- Model ----------------------------

class Net1(nn.Module):   
    def __init__(self):
        super(Net1, self).__init__()

        self.cnn_layers = nn.Sequential(
          # Defining first 2D convolution layer
          nn.Conv2d(1, 4, kernel_size=3, stride=1, padding=1),
          nn.ReLU(inplace=True),
          nn.MaxPool2d(kernel_size=2, stride=2),
          # Defining second 2D convolution layer
          nn.Conv2d(4, 4, kernel_size=3, stride=1, padding=1),
          nn.ReLU(inplace=True),
          nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.linear_layers = nn.Sequential(
          nn.Linear(4 * 7 * 7, 10)
        )

    # Defining the forward pass    
    def forward(self, x):
        x = self.cnn_layers(x)
        x = x.view(x.size(0), -1)
        x = self.linear_layers(x)
        return x

class Net2(nn.Module):
    def __init__(self):
        super(Net2, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4*4*50, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4*4*50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


class Net3(nn.Module):
    def __init__(self):
        super(Net3, self).__init__()
        # Convolution 1
        self.cnn1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=0)
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)
     
        # Convolution 2
        self.cnn2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=0)
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)
        
        # Fully connected 1
        self.fc1 = nn.Linear(32 * 5 * 5, 10) 
    
    def forward(self, x):
        # Set 1
        out = self.cnn1(x)
        out = self.relu1(out)
        out = self.maxpool1(out)
        
        # Set 2
        out = self.cnn2(out)
        out = self.relu2(out)
        out = self.maxpool2(out)
        
        #Flatten
        out = out.view(out.size(0), -1)

        #Dense
        out = self.fc1(out)
        
        return out

# ------------------------ Training/Testing ------------------------

def train_loop(dataloader, model, loss_func, optimizer):
    """Train model using data given by dataloader, with a given loss
    and optimizer.

    Parameters
    ----------
    dataloader : torch.data.Dataloader
        Dataloader serving data to training loop
    model : torch.nn.Module
        Model to train
    loss_func : torch.nn.modules.loss
        Loss function to use in training
    optimizer : torch.optim
        Optimizer to use in training
    """
    for it, (X, y) in enumerate(dataloader):
        
        # Compute prediction and loss
        pred = model(X)            # Forward pass
        loss = loss_func(pred, y)  # Compute the loss

        # Backpropagation
        optimizer.zero_grad()      # Reinitialise the accumulated gradients
        loss.backward()            # Recompute the new gradients
        optimizer.step()           # Update the parameters of the model
        
        # Compute the accuracy on this batch of images
        accuracy_tr = (pred.argmax(1) == y).type(torch.float).sum().item() / len(X)
        
        #if (it+1 == len(dataloader)):
        print('It {}/{}:\tLoss train: {:.5f}, Accuracy train: {:.2%}'.
              format(it + 1, len(dataloader), loss, accuracy_tr), end='\r')
    print()

    
def test_loop(dataloader, model, loss_func):
    """Test given model with data served by dataloader, with a
    given loss function.

    Parameters
    ----------
    dataloader : torch.data.DataLoader
        DataLoader serving data to testing loop
    model : torch.nn.Module
        Model to test
    loss_func : torch.nn.modules.loss
        Loss function to use in testing
    """
    size = len(dataloader.dataset)
    test_loss, accuracy = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_func(pred, y).item()
            accuracy += (pred.argmax(1) == y).type(torch.float).sum().item()

    # Statistics over all the batchs
    test_loss /= size
    accuracy /= size    
    
    print(f"Test Error:\n\tAvg loss: {test_loss:.5f}, Accuracy: {accuracy:.2%}\n")
