import numpy as np
import pandas as pd

from skimage import img_as_ubyte
from skimage.color import rgb2hsv


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


def load_segmented_rank(filepath):
    df_segmented_rank = pd.read_pickle(filepath)
    df_segmented_numbers = df_segmented_rank[~df_segmented_rank['rank'].isin(['Q', 'J', 'K'])].reset_index(drop=True)
    df_segmented_figures = df_segmented_rank[df_segmented_rank['rank'].isin(['Q', 'J', 'K'])].reset_index(drop=True)
    
    return df_segmented_rank, df_segmented_numbers, df_segmented_figures
  

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

def JQK_to_number(letter):
    if letter == "J": return 10
    elif letter == "Q": return 11
    elif letter == "K": return 12
    else: return int(letter)


def count_points_standard(final_df):
    counter = [0,0,0,0]
    for index, row in final_df.iterrows():
        array_of_ranks = final_df[["P1", "P2", "P3", "P4"]].loc[index].apply(lambda x: JQK_to_number(x[0])).values
        max_rank = array_of_ranks.max()
        for i in range(4):
            if array_of_ranks[i] == max_rank:
                counter[i]+=1
    return counter


def count_points_advanced(final_df):
    counter = np.array([0,0,0,0])
    for index, row in final_df.iterrows():  
        player = final_df.loc[index]["D"]
        suit = final_df.loc[index][f"P{player}"][1]

        cards_list = final_df.loc[index][["P1", "P2", "P3", "P4"]].apply(lambda x: (JQK_to_number(x[0]),x[1]))
        filt = cards_list.apply(lambda x: x[1]==suit)

        suit_cards = cards_list[filt]
        best_card = max(suit_cards.values, key=lambda item:item[0])
        
        points = (cards_list == best_card).astype(int).values
        counter = counter + points
    return counter




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
