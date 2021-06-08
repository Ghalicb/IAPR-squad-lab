import numpy as np
import pandas as pd

import cv2
from skimage.util import img_as_ubyte

from skimage.color import rgb2hsv
from skimage.filters import threshold_multiotsu, median
from skimage.measure import label, regionprops_table
from skimage.morphology import (square, binary_opening,
                        binary_closing, opening, binary_dilation, binary_erosion, remove_small_objects)


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
    
    return df_objs.nsmallest(1, "dealer_similarity_measure").iloc[0] #Return a series


#Input: Original image of the round; DataFrame containing the 4 card objects
#Output: Ordered DataFrame (index 0 = player 1; index 1 = player 2; index 2 = player 3; index 3 = player 4)
#that contains the extracted cards (colour and binary version). Extracted cards are also rotated depending on
#the player position

def extract_ordered_cards_from_original_round_image(coloured_image, cards_only_df):
    """Given the image of a round and a dataframe containing the 4 card objects, extract the 4 cards
    from the image by player order (Player 1, 2, 3, 4).

    Parameters
    ----------
    coloured_image : numpy Array
        numpy
    cards_only_df : [type]
        [description]

    Returns
    -------
    [type]
        [description]
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

# ------------------------- Plotting -------------------------

#Input: coloured image of the round
#Output: Series containing a summary of the results (an extracted card for each player + attribution of the dealer) 
def plot_overlay_for_round_image(coloured_image):
    
    df_objs = find_potential_objects_in_original_round_image(coloured_image)
    
    cards_only_df = select_cards_from_potential_objects(df_objs)
    dealer_series = select_dealer_from_potential_objects(df_objs)
    
    ordered_cards_only_df = extract_cards_from_original_round_image(coloured_image, cards_only_df)
    ordered_cards_only_df = add_dealer_status_to_extracted_cards(ordered_cards_only_df, dealer_series)
    
    
    fig = px.imshow(rgb2gray(coloured_image), binary_string=True)
    fig.update_traces(hoverinfo='skip') # hover is only for label info
    properties = ['player', 'dealer']

    # For each label, add a filled scatter trace for its contour,
    # and display the properties of the label in the hover of this trace.
    result_series = pd.Series() # store information of the round. This is the output of the function
    
    for index in range(4):
        result_series[f"P{index+1}_extracted_card"] = (ordered_cards_only_df.iloc[index]["image_coloured"], ordered_cards_only_df.iloc[index]["image"])
        
        label_ = ordered_cards_only_df.iloc[index]["label"]
        contour = find_contours(ordered_cards_only_df.iloc[index]["labeled_im"] == label_, 0.5)[0]
        y, x = contour.T
        hoverinfo = ''
        for prop_name in properties:
            hoverinfo += f'<b>{prop_name}: {ordered_cards_only_df.iloc[index][prop_name]}</b><br>'
        fig.add_trace(go.Scatter(
            x=x, y=y, #name = label_
            mode='lines', fill='toself', showlegend=False,
            hovertemplate=hoverinfo, hoveron='points+fills'))
    
    result_series["dealer"] = int(ordered_cards_only_df[ordered_cards_only_df["dealer"]]["player"][0]) #
    
    label_dealer = dealer_series["label"]
    contour_dealer = find_contours(dealer_series["labeled_im"] == label_dealer, 0.5)[0]
    y_d, x_d = contour_dealer.T
    fig.add_trace(go.Scatter(
        x=x_d, y=y_d, name = "dealer",
        mode='lines', fill='toself', showlegend=False,
        hovertemplate=" ", hoveron='points+fills'))

    fig.show()
    return result_series 
    