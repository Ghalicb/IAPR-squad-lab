import numpy as np
import pandas as pd

import cv2
from skimage.util import img_as_ubyte, crop

from skimage.color import rgb2hsv, rgb2gray

from skimage.transform import resize, rotate

from skimage.measure import label, regionprops_table, find_contours
from skimage.filters import threshold_multiotsu, median, threshold_otsu

from skimage.segmentation import clear_border

from skimage.morphology import (square, binary_opening, binary_closing, opening, 
                binary_dilation, binary_erosion, remove_small_objects)


from utils import rgb_to_hsv, split_img

def preprocess_image(coloured_image):
    """Preprocess image of a round.
    Parameters
    ----------
    coloured_image : numpy Array
        Coloured image of a round
    Returns
    -------
    """
    hsv_img = rgb2hsv(coloured_image) 
    hsv_img = img_as_ubyte(hsv_img) # Convert 0-1 to 0-255 
    
    thres_brightness = threshold_multiotsu(hsv_img[:,:,2]).max() # threshold for the "value" channel

    # HSV image is filtered on the 3 channels:
    # H: [145-200] white colour , S: [0-255] , V: [threshold-255] high values
    filtered_im = cv2.inRange(hsv_img, np.array([145,0,thres_brightness]), np.array([200,255,255])) # Binary filtered image
    
    return filtered_im

def find_potential_objects_in_original_round_image(preprocessed_image):
    """Finds potential objects in image and returns a dataframe containing
    all the segmented objects along with geometrical properties.
    Parameters
    ----------
    preprocessed_image : numpy Array
        Preprocessed image
    Returns
    -------
    pandas DataFrame
        DataFrame containing all segmented objects (cards + dealer + random objects)
        with geometrical properties
    """

    labeled_im = label(preprocessed_image, background=None, connectivity=preprocessed_image.ndim) # Find objects using region growing and labeling
    labeled_im = remove_small_objects(labeled_im, min_size=30000) # Remove objects smaller than the dealer and the cards
    labeled_im = median(labeled_im, square(15)) # Remove noise
    
    # Dictionnary that stores information for each segmented objects 
    props_objs = regionprops_table(labeled_im, properties=(
                                                            "label",
                                                            "area",
                                                            "filled_area",
                                                            "major_axis_length",
                                                            "minor_axis_length",
                                                            'centroid',
                                                            "slice",
                                                            "image"
                                                            ))
    df_objs = pd.DataFrame(props_objs)
    df_objs["labeled_im"] = 0 # random value
    df_objs["labeled_im"] = df_objs["labeled_im"].apply(lambda x: labeled_im)
    
    return df_objs


def select_cards_from_potential_objects(df_objs):
    """Given a dataframe containing all segmented objects of an image,
    select only cards using a similarity measure.
    Parameters
    ----------
    df_objs : pandas DataFrame
        DataFrame contaning all the potential objects along with
        their geometrical properties
    Returns
    -------
    pandas DataFrame
        DataFrame with 4 objects corresponding to the 4 cards in the
        image, with no specific ordering
    """
    
    # Features differentiating cards from the rest of the objects
    # These are reference values obtained by exploring the dataset
    C_REF_MAJOR_AXIS_MEAN = 838
    C_REF_MAJOR_AXIS_STD = 16
    C_REF_MINOR_AXIS_MEAN = 542
    C_REF_MINOR_AXIS_STD = 23
    
    cards_feature1_series = (df_objs["major_axis_length"]-C_REF_MAJOR_AXIS_MEAN)/C_REF_MAJOR_AXIS_STD # Normalized feature 1 
    cards_feature2_series = (df_objs["minor_axis_length"]-C_REF_MINOR_AXIS_MEAN)/C_REF_MINOR_AXIS_STD # Normalized feature 2
    
    # Euclidian distance corresponding to the similarity between an object and the reference card.
    # The smaller the distance, the greater the similarity. Here the 4 most similar objects are kept.
    cards_similarity_distances_series = (cards_feature1_series**2 + cards_feature2_series**2) ** 0.5 # Euclidian distance
    df_objs["card_similarity_measure"] = cards_similarity_distances_series
    
    return df_objs.nsmallest(4, "card_similarity_measure")


def select_dealer_from_potential_objects(df_objs): 
    """Given a dataframe containing all segmented objects of an image,
    select only the dealer chip using a similarity measure.
    Parameters
    ----------
    df_objs : pandas DataFrame
        DataFrame contaning all the potential objects along with
        their geometrical properties
    Returns
    -------
    pandas Series
        Series with 1 object corresponding to the dealer chip
    """
    
    # Features differentiating the dealer from the rest of the objects
    # These are reference values obtained by exploring the dataset
    D_REF_AREA_MEAN = 45061
    D_REF_AREA_STD = 1202
    D_REF_FILLED_AREA_MEAN = 65325
    D_REF_FILLED_AREA_STD = 1134

    dealer_feature1_series = (df_objs["area"]-D_REF_AREA_MEAN)/D_REF_AREA_STD # Normalized feature 1 
    dealer_feature2_series = (df_objs["filled_area"]-D_REF_FILLED_AREA_MEAN)/D_REF_FILLED_AREA_STD # Normalized feature 2
    
    # Euclidian distance corresponding to the similarity between an object and the reference dealer.
    # The smaller the distance, the greater the similarity. Here the most similar object is kept.
    dealer_similarity_distances_series = (dealer_feature1_series**2+dealer_feature2_series**2)**0.5 # Euclidian distance
    df_objs["dealer_similarity_measure"] = dealer_similarity_distances_series
    
    return df_objs.nsmallest(1, "dealer_similarity_measure").iloc[0]


def extract_ordered_cards(coloured_image, cards_only_df):
    """Given the image of a round and a dataframe containing the 4 card objects, extract the 4 cards
    from the image by player order (Player 1, 2, 3, 4).
    Parameters
    ----------
    coloured_image : np.array
        Coloured image of a round
    cards_only_df : pd.DataFrame    
        DataFrame containing only cards and their geometrical
        properties
    Returns
    -------
    pd.DataFrame
        Ordered DataFrame (index 0 = player 1; index 1 = player 2; index 2 = player 3; index 3 = player 4)
        that contains the extracted cards (colour and binary version). Extracted cards are also rotated depending on
        the player position
    """

    ordered_cards_only_df = pd.DataFrame()
    
    # Players are found using the centroid of the card object. Centroid = (x0, y0)
    player1_series = cards_only_df.loc[cards_only_df["centroid-0"].idxmax()] # Player 1 has the highest x0
    
    # The "slice" property is used to extract the card from the original image
    player1_series["image_coloured"] = coloured_image[player1_series["slice"]] 
   
    player2_series = cards_only_df.loc[cards_only_df["centroid-1"].idxmax()] # Player 2 has the highest y0
    
    # The "image" property is used to extract the card from the original image in binary version
    player2_series["image"] = rotate(player2_series["image"], -90, resize=True) # Rotation of the extracted binary card
    player2_series["image_coloured"] = rotate(coloured_image[player2_series["slice"]], -90, resize=True)

    player3_series = cards_only_df.loc[cards_only_df["centroid-0"].idxmin()] # Player 3 has the lowest x0
    player3_series["image"] = rotate(player3_series["image"], 180, resize=True)
    player3_series["image_coloured"] = rotate(coloured_image[player3_series["slice"]], 180, resize=True)

    player4_series = cards_only_df.loc[cards_only_df["centroid-1"].idxmin()] # Player 4 has the lowest y0
    player4_series["image"] = rotate(player4_series["image"], 90, resize=True)
    player4_series["image_coloured"] = rotate(coloured_image[player4_series["slice"]], 90, resize=True)


    ordered_cards_only_df = (ordered_cards_only_df.append(player1_series, ignore_index=True)
                             .append(player2_series, ignore_index=True)
                             .append(player3_series, ignore_index=True)
                             .append(player4_series, ignore_index=True))
    
    ordered_cards_only_df["player"] = [1,2,3,4]
    
    return ordered_cards_only_df


def add_dealer_status_to_extracted_cards(ordered_cards_only_df, dealer_series):
    """Given the DataFrame containing cards ordered by player number, and the series
    containing information about the dealer chip, gives the dealer status to the
    appropriate player.

    Parameters
    ----------
    ordered_cards_only_df : pd.DataFrame
        DataFrame containing information about cards
        ordered by player number
    dealer_series : pd.Series
        Series containing information about the dealer chip

    Returns
    -------
    pd.DataFrame
        DataFrame containing cards ordered by player number, with a "dealer" column,
        indicating whether a player is the dealer or not
    """
    
    dealer_centr0 = dealer_series["centroid-0"]
    dealer_centr1 = dealer_series["centroid-1"]
   
    # To find the dealer, the euclidian distance between the centroids of the cards and the dealer is computed 
    ordered_cards_only_df["dealer"] = [False,False,False,False]
    distances_to_dealer_series = ((ordered_cards_only_df["centroid-0"]-dealer_centr0)**2 + 
                                    (ordered_cards_only_df["centroid-1"]-dealer_centr1)**2)**0.5 # Euclidian distance
    ordered_cards_only_df.loc[distances_to_dealer_series.idxmin(),"dealer"] = True
    
    return ordered_cards_only_df


def pipeline_segmentation_round(coloured_image):
    """Pipeline of segmentation of a round, extracting the cards and
    the dealer chip along with their geometrical properties.

    Parameters
    ----------
    coloured_image : np.array
        Coloured image of a round

    Returns
    -------
    (pd.Series, pd.DataFrame, pd.Series)
        Series with information about the dealer, a DataFrame containing information
        about the cards ordered by player number, and a Series containing a summary about
        the different objects to be extract from the image (extracted image + contour)
    """
    df_objs = find_potential_objects_in_original_round_image(coloured_image)
    
    cards_only_df = select_cards_from_potential_objects(df_objs)
    dealer_series = select_dealer_from_potential_objects(df_objs)
    
    ordered_cards_only_df = extract_ordered_cards(coloured_image, cards_only_df)
    ordered_cards_only_df = add_dealer_status_to_extracted_cards(ordered_cards_only_df, dealer_series)

    result_series = pd.Series()

    for index in range(4):
        result_series[f"P{index+1}_extracted_card"] = (ordered_cards_only_df.iloc[index]["image_coloured"], \
                                                        ordered_cards_only_df.iloc[index]["image"])
        label_ = ordered_cards_only_df.iloc[index]["label"]
        contour = find_contours(ordered_cards_only_df.iloc[index]["labeled_im"] == label_, 0.5)[0]
        result_series[f"P{index+1}_extracted_card_contour"] = contour
    
    result_series["dealer"] = int(ordered_cards_only_df[ordered_cards_only_df["dealer"]]["player"][0])
    
    label_dealer = dealer_series["label"]
    contour_dealer = find_contours(dealer_series["labeled_im"] == label_dealer, 0.5)[0]
    result_series[f"dealer_contour"] = contour_dealer

    return dealer_series, ordered_cards_only_df, result_series


def extract_rank_from_card(coloured_card, binary_card, backup_image):
    """Extracts rank from segmented card and its binary equivalent. If no
    rank is found, return a backup image.

    Parameters
    ----------
    coloured_card : np.array
        Segmented coloured card
    binary_card : np.array
        Segmented card in binary format
    backup_image : np.array
        Backup rank to return in case no rank
        is found

    Returns
    -------
    np.array
        Cropped gray-scaled image of the rank, scaled-down to 28x28
    """

    crop_factor = 0.15

    cropped_binary_card = crop(binary_card, ((binary_card.shape[0]*crop_factor, binary_card.shape[0]*crop_factor), 
                                                (binary_card.shape[1]*crop_factor, binary_card.shape[1]*crop_factor)))
    cropped_coloured_card = crop(coloured_card, ((coloured_card.shape[0]*crop_factor, coloured_card.shape[0]*crop_factor), 
                                                    (coloured_card.shape[1]*crop_factor, coloured_card.shape[1]*crop_factor), (0, 0)))

    # Take the negative of the segmented binary card (numbers become objects instead of background)
    filt1 = (cropped_binary_card == False)
    filt2 = (cropped_binary_card == True)
    cropped_binary_card[filt1] = True
    cropped_binary_card[filt2] = False   
    cropped_binary_card = binary_dilation(cropped_binary_card, square(45)) # Closes the number having holes i.e some 5 and 2
    labeled_im = remove_small_objects(label(cropped_binary_card), 5000)
    
    props_objs = regionprops_table(labeled_im, properties=(
                                                  "label",
                                                  "area",
                                                  'centroid',"slice"))
    df_objs = pd.DataFrame(props_objs)
    
    if(len(df_objs)==0): 
        print("A Rank not detected. backup Queen was put instead...")
        return backup_image
    
    else:
        coloured_cropped_number = cropped_coloured_card[df_objs.loc[df_objs["area"].idxmax()]["slice"]] # extract the number (biggest object)
        cropped_number_gray = rgb2gray(coloured_cropped_number)
        thresh = threshold_otsu(cropped_number_gray)
        mnist_im = resize(img_as_ubyte(cropped_number_gray < thresh), (28,28))
        return mnist_im
 

def extract_suit_from_card(coloured_card, backup_image, upper_left=True):
    """Extracts the suit from a segmented coloured card. If no suit is
    detected, return a backup image.

    Parameters
    ----------
    coloured_card : np.array
        Segmented coloured card
    backup_image : np.array
        Backup suit to return in case no
        suit is found
    upper_left : bool, optional
        Whether to take the suit in the upper left corner
        or lower right, by default True

    Returns
    -------
    np.array
        Cropped gray-scaled image of the suit, scaled-down to 28x28
    """
    if (upper_left):
        cropped_coloured_card = crop(coloured_card, ((0, coloured_card.shape[0]*0.6), (0, coloured_card.shape[1]*0.5), (0, 0)))
    else:
        cropped_coloured_card = crop(coloured_card, ((coloured_card.shape[0]*0.6, 0), (coloured_card.shape[1]*0.5, 0), (0, 0)))

    cropped_coloured_card_gray = rgb2gray(cropped_coloured_card)
    thresh = threshold_otsu(cropped_coloured_card_gray)
    cropped_coloured_card_gray_binary = img_as_ubyte(cropped_coloured_card_gray < thresh)
    labeled_im = label(cropped_coloured_card_gray_binary)
    labeled_im = clear_border(labeled_im)
    
    props_objs = regionprops_table(labeled_im, properties=(
                                                  "label",
                                                  "area",
                                                  'centroid',"slice"))
    df_objs = pd.DataFrame(props_objs)

    if(len(df_objs)==0): 
        print("A suit not detected. backup Spade was put instead... ")
        return backup_image
    else:
        extracted_suit_coloured = cropped_coloured_card[df_objs.loc[df_objs["area"].idxmax()]["slice"]]

        extracted_suit_gray = rgb2gray(extracted_suit_coloured)
        thresh = threshold_otsu(extracted_suit_gray)
        return resize(img_as_ubyte(extracted_suit_gray < thresh), (28,28))


def extract_all_ranks_suits(dict_data, nb_games, nb_rounds, backup_rank, backup_suit):
    """Extract all ranks and suits from the images of rounds across games.

    Parameters
    ----------
    dict_data : dict
        Dictionary containing the images of rounds,
        categorized by game number
    nb_games : int
        Number of games to extract from
    nb_rounds : int
        Number of rounds to extract from
    backup_rank : np.array
        Backup rank in case no rank is found
        during extraction
    backup_suit : np.array
        Backup suit in case no suit is found
        during extraction

    Returns
    -------
    (pd.DataFrame, pd.DataFrame)
        A tuple of DataFrames, the first one containing all the ranks and the second
        one all the suits
    """
    df_ranks = []
    df_suits = []
    for game_nb in range(1, nb_games+1):
        for i in range(1, nb_rounds+1):
            coloured_image = dict_data[f'game{game_nb}'][f'round{i}']
            df_ground_truth = pd.read_csv(f"data/train_games/game{game_nb}/game{game_nb}.csv")
            
            preprocessed_img = preprocess_image(coloured_image)
            df_objs = find_potential_objects_in_original_round_image(preprocessed_img)
            cards_only_df = select_cards_from_potential_objects(df_objs)      
            cards_ordered_df = extract_ordered_cards(coloured_image, cards_only_df)
            
            for player_nb in range(4):
                true_rank = df_ground_truth.loc[i-1, f"P{player_nb+1}"][0]
                true_suit = df_ground_truth.loc[i-1, f"P{player_nb+1}"][1]
                
                coloured_card = cards_ordered_df.loc[player_nb]["image_coloured"]
                binary_card = cards_ordered_df.loc[player_nb]["image"]
                rank = extract_rank_from_card(coloured_card, binary_card, backup_rank)
                suit_upper_left = extract_suit_from_card(coloured_card, backup_suit, upper_left=True)
                suit_lower_right = extract_suit_from_card(coloured_card, backup_suit, upper_left=False)
                
                df_ranks.append({'rank': true_rank, 'suit': true_suit, 'image': rank})
                df_suits.append({'rank': true_rank, 'suit': true_suit, 'image': suit_upper_left})
                df_suits.append({'rank': true_rank, 'suit': true_suit, 'image': rotate(suit_lower_right, 180)})
            
    df_ranks = pd.DataFrame(df_ranks)
    df_suits = pd.DataFrame(df_suits)
    
    return df_ranks, df_suits


# -------------------------- EDGE-BASED --------------------------
def find_sorted_contours(img):
    """Given an image, find all the contours in the image
    and sort them by area covered by the contour.

    Parameters
    ----------
    img : np.array
        Image to perform contour detection on

    Returns
    -------
    np.array
        Contours sorted by area
    """
    # Find the contours of our image
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Sort the contours by area covered within the contours
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    
    return contours

def detect_dealer_chip(img):
    """Detect dealer chip in image using edge-based method.

    Parameters
    ----------
    img : np.array
        Image to extract the dealer chip from

    Returns
    -------
    (np.array, np.array, np.array, int, np.array)
        The contour of the chip, the centers of the minimum enclosing circle
        and its corresponding radius, as well as the preprocessed image
    """
    # Convert RGB to HSV image
    hsv_img = rgb_to_hsv(img)
    
    # Split the image
    hue_img, sat_img, val_img = split_img(hsv_img)
    
    # Threshold the image and transform it to a binary image
    img_processed = (hue_img > 60) & (hue_img < 120) & (sat_img > 90) & (sat_img < 180)
    
    # Convert the image from binary to [0-255]
    img_processed = img_as_ubyte(img_processed)
    
    # Apply a closing to the image (close the holes in the dealer chip)
    kernel_ell = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(15,15))
    img_processed = cv2.morphologyEx(img_processed, cv2.MORPH_CLOSE, kernel_ell)

    # Apply an opening to the image (remove the cards)
    kernel_rec = cv2.getStructuringElement(cv2.MORPH_RECT,(45,45))
    img_processed = cv2.morphologyEx(img_processed, cv2.MORPH_OPEN, kernel_rec)
    
    # Find the contour & the enclosing circle of the chip
    contour_chip = find_sorted_contours(img_processed)[0]
                
    contour_chip_poly = cv2.approxPolyDP(contour_chip, 3, True)
    centers, radius = cv2.minEnclosingCircle(contour_chip_poly)
    
    return contour_chip, centers, radius, img_processed