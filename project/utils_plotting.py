import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from skimage.color import rgb2gray, rgb2hsv
from skimage.util import img_as_ubyte

from utils_segmentation import pipeline_segmentation_round


def plot_overlay_for_round_image(coloured_image, ordered_cards_only_df, dealer_series, result_series):
    """Plot overlay for image of a round indicating the cards belonging to each player and the dealer
    chip.

    Parameters
    ----------
    coloured_image : np.array
        Coloured image of a round
    ordered_cards_only_df : pd.DataFrame
        DataFrame containing information about cards
        ordered by player number
    dealer_series : pd.Series
        Series containing information about the dealer chip
    result_series : pd.Series
        Series containing information about the different objects to plot
        on the overlay (image, contour)
    """
    fig = px.imshow(rgb2gray(coloured_image), binary_string=True)
    fig.update_traces(hoverinfo='skip') # hover is only for label info
    properties = ['player', 'dealer']

    # For each label, add a filled scatter trace for its contour,
    # and display the properties of the label in the hover of this trace.
    result_series = pd.Series() # store information of the round. This is the output of the function
    
    for index in range(4):
        label_ = ordered_cards_only_df.iloc[index]["label"]
        contour = result_series[f"P{index+1}_extracted_card_contour"]
        y, x = contour.T
        hoverinfo = ''
        for prop_name in properties:
            hoverinfo += f'<b>{prop_name}: {ordered_cards_only_df.iloc[index][prop_name]}</b><br>'
        fig.add_trace(go.Scatter(
            x=x, y=y, #name = label_
            mode='lines', fill='toself', showlegend=False,
            hovertemplate=hoverinfo, hoveron='points+fills'))
    
    label_dealer = dealer_series["label"]
    contour_dealer = result_series["dealer_contour"]
    y_d, x_d = contour_dealer.T
    fig.add_trace(go.Scatter(
        x=x_d, y=y_d, name = "dealer",
        mode='lines', fill='toself', showlegend=False,
        hovertemplate=" ", hoveron='points+fills'))

    fig.show()


def pipeline_segment_plot_game(dict_data, game_number=1):
    """Plot overlay for all rounds of a game and return the results of the
    segmentation.

    Parameters
    ----------
    dict_data : dict
        Dictionary containing all the images of a game
    game_number : int, optional
        Number of the game, by default 1

    Returns
    -------
    pd.DataFrame
        Results of segmentation of each round.
    """
    results_game = pd.DataFrame()
    for i in range(13):
        coloured_image = dict_data[f"game{game_number}"][f"round{i+1}"]
        print(f"The segmentation of cards and dealer for round {i+1} is:")

        dealer_series, ordered_cards_only_df, result_series = pipeline_segmentation_round(coloured_image)
        plot_overlay_for_round_image(coloured_image, ordered_cards_only_df, dealer_series, result_series)
        results_game = results_game.append(dealer_series, ignore_index=True)
    
    return results_game


def plot_round_image_in_HSV(coloured_image):
    """Given a coloured image of a round, plot the corresponding
    HSV channels.

    Parameters
    ----------
    coloured_image : np.array
        Coloured image of a round
    """
    hsv_img = rgb2hsv(coloured_image) 
    hsv_img = img_as_ubyte(hsv_img) # Convert 0-1 to 0-255 
    fig = px.imshow(hsv_img, binary_string=True, title="HSV channels of the image")
    fig.show()