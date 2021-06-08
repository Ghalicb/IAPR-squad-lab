import numpy as np

import torch

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
