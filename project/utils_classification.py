import os
import gzip
import tarfile

import pickle
import pandas as pd

import numpy as np
import matplotlib.pyplot as plt

import cv2
from skimage import img_as_ubyte

# Torch
import torch
from torch import nn
import torch.nn.functional as F

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
    imgs = [v[1] for v in values]
    ranks = [v[0][0] for v in values]
    suits = [v[0][1] for v in values]

    df_task0 = pd.DataFrame({"rank": ranks, "suit": suits, "image": imgs})
    df_numbers_only = df_task0[~df_task0['rank'].isin(['Q', 'J', 'K'])].reset_index(drop=True)
    df_not_numbers = df_task0[df_task0['rank'].isin(['Q', 'J', 'K'])].reset_index(drop=True)
    
    return df_task0, df_numbers_only, df_not_numbers
  

def create_figures_dataset(imgs, fig_label, labels_figures, transform, nb_samples):
    """Augments the array of figures using a given transform, until the number of 
    samples in dataset reaches nb_samples.

    Parameters
    ----------
    imgs : Numpy ndarray
        Images of figures to augment
    fig_label : str
        Label of the figure we are augmenting
    labels_figures : dict
        Dictionary mapping each figure label (J, Q, K) to a number
    transform : torchvision.transforms.Compose
        A composition of transforms to apply on images
    nb_samples : int
        Number of samples in the final dataset

    Returns
    -------
    (Numpy ndarray, Numpy ndarray)
        A tuple containing the augmented dataset and its corresponding array of labels
    """
    size_dataset = imgs.shape[0]
    image_shape = imgs[0].shape
    iterations = int(nb_samples/imgs.shape[0])
    
    label = labels_figures[fig_label]
    augmented_data = []
    labels = []
    for i in range(iterations):
        for img in imgs:
            augmented = img_as_ubyte(transform(img)).reshape(image_shape)
            augmented_data.append(augmented)
            labels.append(label)
    
    return np.array(augmented_data), np.array(labels)

def get_train_val_test_figures(df_not_numbers, labels_figures, transform):
    """Given the dataframe containing figures (Jack, Queen, King) and the
    dictionary mapping the label of a figure to an int, create a train,
    validation and test set for figures to use in training and testing
    a model. Train and validation sets are created using augmentation to
    get a similar number of samples to classes in MNIST.

    Parameters
    ----------
    df_not_numbers : pandas DataFrame
        Dataframe containing figures and their images
    labels_figures : dict
        Dictionary mapping labels of figures to ints
    transform : torchvision.transforms.Compose
        A composition of transforms to apply on images

    Returns
    -------
    (Numpy ndarray, Numpy ndarray, Numpy ndarray, Numpy ndarray, Numpy ndarray, Numpy ndarray)
        Arrays corresponding to train, validation and test sets with their
        corresponding labels
    """
    kings = df_not_numbers[df_not_numbers['rank'] == 'K'].reset_index(drop=True)
    queens = df_not_numbers[df_not_numbers['rank'] == 'Q'].reset_index(drop=True)
    jacks = df_not_numbers[df_not_numbers['rank'] == 'J'].reset_index(drop=True)

    kings_imgs = kings.image.apply(img_as_ubyte)
    queens_imgs = queens.image.apply(img_as_ubyte)
    jacks_imgs = jacks.image.apply(img_as_ubyte)
    
    # Train
    train_kings = kings_imgs[:int(0.7*kings.shape[0])]
    train_queens = queens_imgs[:int(0.7*queens.shape[0])]
    train_jacks = jacks_imgs[:int(0.7*jacks.shape[0])]
    
    train_kings_aug, kings_labels = create_figures_dataset(train_kings, 'K', labels_figures, transform, 5000)
    train_queens_aug, queens_labels = create_figures_dataset(train_queens, 'Q', labels_figures, transform, 5000)
    train_jacks_aug, jacks_labels = create_figures_dataset(train_jacks, 'J', labels_figures, transform, 5000)
    
    train_augmented_figs = np.concatenate((train_kings_aug, train_queens_aug, train_jacks_aug))
    train_augmented_figs_labels = np.concatenate((kings_labels, queens_labels, jacks_labels))
    
    # Validation 
    val_kings = kings_imgs[int(0.7*kings.shape[0]):int(0.8*kings.shape[0])].reset_index(drop=True)
    val_queens = queens_imgs[int(0.7*queens.shape[0]):int(0.8*queens.shape[0])].reset_index(drop=True)
    val_jacks = jacks_imgs[int(0.7*jacks.shape[0]):int(0.8*jacks.shape[0])].reset_index(drop=True)

    val_kings, val_kings_labels = create_figures_dataset(val_kings, 'K', labels_figures, transform, 500)
    val_queens, val_queens_labels = create_figures_dataset(val_queens, 'Q', labels_figures, transform, 500)
    val_jacks, val_jacks_labels = create_figures_dataset(val_jacks, 'J', labels_figures, transform, 500)

    val_augmented_figs = np.concatenate((val_kings, val_queens, val_jacks))
    val_augmented_figs_labels = np.concatenate((val_kings_labels, val_queens_labels, val_jacks_labels))
    
    
    # Test
    test_kings = np.array([v for v in kings_imgs[int(0.8*kings.shape[0]):]])
    test_queens = np.array([v for v in queens_imgs[int(0.8*queens.shape[0]):]])
    test_jacks = np.array([v for v in jacks_imgs[int(0.8*jacks.shape[0]):]])
    
    test_figs = np.concatenate((test_kings, test_queens, test_jacks))
    test_figs_labels = np.array([labels_figures['K']]*test_kings.shape[0] + [labels_figures['Q']]*test_queens.shape[0] \
                                    + [labels_figures['J']]*test_jacks.shape[0])
    
    return train_augmented_figs, train_augmented_figs_labels, val_augmented_figs, val_augmented_figs_labels, \
            test_figs, test_figs_labels


def get_train_val_test_suits(df_suits, labels_suits, transform):
    """Given the dataframe containing suits and the dictionary mapping 
    the label of a figure to an int, create a train, validation and test 
    set for figures to use in training and testing a model. Train and 
    validation sets are created using augmentation to get a to a certain
    number of samples.

    Parameters
    ----------
    df_suits : pandas DataFrame
        Dataframe containing suits and their images
    labels_suits : dict
        Dictionary mapping labels of suits to ints
    transform : torchvision.transforms.Compose
        A composition of transforms to apply on images

    Returns
    -------
    (Numpy ndarray, Numpy ndarray, Numpy ndarray, Numpy ndarray, Numpy ndarray, Numpy ndarray)
        Arrays corresponding to train, validation and test sets with their
        corresponding labels
    """
    df_clubs = df_suits[df_suits['suit'] == 'C'].reset_index(drop=True).image.apply(img_as_ubyte)
    df_diamonds = df_suits[df_suits['suit'] == 'D'].reset_index(drop=True).image.apply(img_as_ubyte)
    df_hearts = df_suits[df_suits['suit'] == 'H'].reset_index(drop=True).image.apply(img_as_ubyte)
    df_spades = df_suits[df_suits['suit'] == 'S'].reset_index(drop=True).image.apply(img_as_ubyte)
    
    # Train
    train_clubs = df_clubs[:int(0.7*df_clubs.shape[0])]
    train_diamonds = df_diamonds[:int(0.7*df_diamonds.shape[0])]
    train_hearts = df_hearts[:int(0.7*df_hearts.shape[0])]
    train_spades = df_spades[:int(0.7*df_spades.shape[0])]
    
    train_clubs_aug, train_clubs_labels = create_figures_dataset(train_clubs, 'C', labels_suits, transform, 1000)
    train_diamonds_aug, train_diamonds_labels = create_figures_dataset(train_diamonds, 'D', labels_suits, transform, 1000)
    train_hearts_aug, train_hearts_labels = create_figures_dataset(train_hearts, 'H', labels_suits, transform, 1000)
    train_spades_aug, train_spades_labels = create_figures_dataset(train_spades, 'S', labels_suits, transform, 1000)
    
    train_augmented_suits = np.concatenate((train_clubs_aug, train_diamonds_aug, train_hearts_aug, train_spades_aug))
    train_augmented_suits_labels = np.concatenate((train_clubs_labels, train_diamonds_labels, train_hearts_labels, train_spades_labels))
    
    # Validation 
    val_clubs = df_clubs[int(0.7*df_clubs.shape[0]):int(0.8*df_clubs.shape[0])].reset_index(drop=True)
    val_diamonds = df_diamonds[int(0.7*df_diamonds.shape[0]):int(0.8*df_diamonds.shape[0])].reset_index(drop=True)
    val_hearts = df_hearts[int(0.7*df_hearts.shape[0]):int(0.8*df_hearts.shape[0])].reset_index(drop=True)
    val_spades = df_spades[int(0.7*df_spades.shape[0]):int(0.8*df_spades.shape[0])].reset_index(drop=True)

    val_clubs_aug, val_clubs_labels = create_figures_dataset(val_clubs, 'C', labels_suits, transform, 100)
    val_diamonds_aug, val_diamonds_labels = create_figures_dataset(val_diamonds, 'D', labels_suits, transform, 100)
    val_hearts_aug, val_hearts_labels = create_figures_dataset(val_hearts, 'H', labels_suits, transform, 100)
    val_spades_aug, val_spades_labels = create_figures_dataset(val_spades, 'S', labels_suits, transform, 100)

    val_augmented_suits = np.concatenate((val_clubs_aug, val_diamonds_aug, val_hearts_aug, val_spades_aug))
    val_augmented_suits_labels = np.concatenate((val_clubs_labels, val_diamonds_labels, val_hearts_labels, val_spades_labels))
    
    
    # Test
    test_clubs = np.array([v for v in df_clubs[int(0.8*df_clubs.shape[0]):]])
    test_diamonds = np.array([v for v in df_diamonds[int(0.8*df_diamonds.shape[0]):]])
    test_hearts = np.array([v for v in df_hearts[int(0.8*df_hearts.shape[0]):]])
    test_spades = np.array([v for v in df_spades[int(0.8*df_spades.shape[0]):]])
    
    test_suits = np.concatenate((test_clubs, test_diamonds, test_hearts, test_spades))
    test_suits_labels = np.array([labels_suits['C']]*test_clubs.shape[0] + [labels_suits['D']]*test_diamonds.shape[0] \
                                    + [labels_suits['H']]*test_hearts.shape[0] + [labels_suits['S']]*test_spades.shape[0])
    
    return train_augmented_suits, train_augmented_suits_labels, val_augmented_suits, val_augmented_suits_labels, \
            test_suits, test_suits_labels



# ---------------------------- Model ----------------------------

class Net(nn.Module):
    def __init__(self, nb_classes=13):
        super(Net, self).__init__()

        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)

        self.fc1 = nn.Linear(4*4*50, 500)
        self.fc2 = nn.Linear(500, nb_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)

        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)

        x = x.view(-1, 4*4*50)
        x = F.relu(self.fc1(x))

        x = self.fc2(x)

        return F.log_softmax(x, dim=1)

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


def predict(imgs, nb_classes, model_class, model_path):
    """Given an array of 28x28 images, predict the rank or suit of each image using
    the model given as argument.

    Parameters
    ----------
    imgs : Numpy ndarray
        Images to predict the rank or suit on
    model_class : torch.nn.Module
        Class of model to use for predictions
    model_path : str
        Path to model weights

    Returns
    -------
    torch.Tensor
        Tensor containing the predictions
    """
    model = model_class(nb_classes=nb_classes)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    input_model = torch.Tensor(np.expand_dims(imgs, 1))
    preds = model(input_model).argmax(1)
    
    return preds
