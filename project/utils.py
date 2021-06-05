import numpy as np
import matplotlib.pyplot as plt

import cv2
from skimage.color import rgb2hsv
from skimage.exposure import histogram
from skimage.util import img_as_ubyte


def rgb_to_hsv(rgb_img):
    """
    Convert an RGB image into HSV.

    Parameters
    ----------
    rgb_img: 3darray [HxWx3]
        Image with RGB channels.
        
    Returns
    -------
    hsv_img: 3darray [HxWx3]
        Image with HSV channels with values between [0-255].
    """
    # Convert from rgb to hsv
    hsv_img = rgb2hsv(rgb_img)
    
    # Convert from range 0-1 to 0-255
    hsv_img = img_as_ubyte(hsv_img)
    
    return hsv_img
    
    
def split_img(img):
    """
    Split the three channels of an image of dimensions [HxWxC].

    Parameters
    ----------
    img: 3darray [HxWx3]
        Image to split.
       
    Returns
    -------
    tuple of 3 [HxW] array
        A tuple containing each channel of the image
    """
    return (img[:,:,0], img[:,:,1], img[:,:,2])
    
    
def plot_histogram(img, img_type):
    """
    Plot the histogram of each channel of an RGB or HSV image.

    Parameters
    ----------
    img: 3darray [HxWx3]
        Image to plot the histograms.
    img_type: enum ['RGB', 'HSV']
        The type of the image.
    """
    # Split each channel of the hsv image
    channel1, channel2, channel3 = split_img(img)
    
    # Adapt the histogram titles depending on img_type
    title = None
    if (img_type == 'RGB'):
        title = ('Historgram for the Red', 'Historgram for the Green', 'Historgram for the Blue')
    elif (img_type == 'HSV'):
        title = ('Histogram of the Hue', 'Histogram of the Saturation', 'Histogram of the Value')
    
    # Plot the histogram for each channel
    fig, ax = plt.subplots(1, 3, figsize=(20,4), sharex=True, sharey=True)

    hist1, _ = histogram(channel1)
    ax[0].plot(hist1)
    ax[0].set_title(title[0])
    ax[0].set_ylabel("Number of pixels")

    hist2, _ = histogram(channel2)
    ax[1].plot(hist2)
    ax[1].set_title(title[1])

    hist3, _ = histogram(channel3)
    ax[2].plot(hist3)
    ax[2].set_title(title[2])

    plt.show()
    

def evaluate_game(pred, cgt, mode_advanced=False):
    """
    Evalutes the accuracy of your predictions. The same function will be used to assess the 
    performance of your model on the final test game.


    Parameters
    ----------
    pred: array of string of shape NxD
        Prediction of the game. N is the number of round (13) and D the number of players (4). Each row 
        is composed of D string. Each string can is composed of 2 charcters [0-9, J, Q, K] + [C, D, H, S].
        If the mode_advanced is False only the rank is evaluated. Otherwise, both rank and colours are 
        evaluated (suits).
    cgt: array of string of shape NxD
        Ground truth of the game. Same format as the prediciton.
    mode_advanced: bool, optional
        Choose the evaluation mode
        
    Returns
    -------
    accuracy: float
        Accuracy of the prediciton wrt the ground truth. Number of correct entries divided by 
        the total number of entries.
    """
    if pred.shape != cgt.shape:
        raise Exception("Prediction and ground truth sould have same shape.")
    
    if mode_advanced:
        # Full performance of the system. Cards ranks and colours.
        return (pred == cgt).mean()
    else:
        # Simple evaluation based on cards ranks only
        cgt_simple = np.array([v[0] for v in cgt.flatten()]).reshape(cgt.shape)
        pred_simple = np.array([v[0] for v in pred.flatten()]).reshape(pred.shape)
        return (pred_simple == cgt_simple).mean()
    
    
def print_results(rank_colour, dealer, pts_standard, pts_advanced):
    """
    Print the results for the final evaluation. You NEED to use this function when presenting the results on the 
    final exam day.
    
    Parameters
    ----------
    rank_colour: array of string of shape NxD
        Prediction of the game. N is the number of round (13) and D the number of players (4). Each row 
        is composed of D string. Each string can is composed of 2 charcters [0-9, J, Q, K] + [C, D, H, S].
    dealer: list of int
        Id ot the players that were selected as dealer ofr each round.
    pts_standard: list of int of length 4
        Number of points won bay each player along the game with standard rules.
    pts_advanced: list of int of length 4
        Number of points won bay each player along the game with advanced rules.
    """
    print('The cards played were:')
    print(pp_2darray(rank_colour))
    print('Players designated as dealer: {}'.format(dealer))
    print('Players points (standard): {}'.format(pts_standard))
    print('Players points (advanced): {}'.format(pts_advanced))
    
    
def pp_2darray(arr):
    """
    Pretty print array
    """
    str_arr = "[\n"
    for row in range(arr.shape[0]):
        str_arr += '[{}], \n'.format(', '.join(["'{}'".format(f) for f in arr[row]]))
    str_arr += "]"
    return str_arr
