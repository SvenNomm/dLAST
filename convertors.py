# set of function to perform conversion between different coordinate systems
import numpy as np
from sklearn import preprocessing
import pandas as pd



# this function converts np.array representing fine motor test drawing into descartes coordinates.
def convert_to_descartes_coordinates_from_iPad(data, screen_y_max):
    data['y'] = screen_y_max - data['y']
    return data


def iPad2mm(data, x_mm2pix, y_mm2pix):
    data['x'] = data['x'] / x_mm2pix
    data['y'] = data['y'] / y_mm2pix
    return data


def normalize(data):
    for column in data.columns.values:
        if column != 't':
            x = data[column].values  # returns a numpy array
            normalized_x = (x - x.mean()) / x.std()
            #min_max_scaler = preprocessing.MinMaxScaler()

            data[column] = normalized_x
    return data


def minmax_normalize(data):
    for column in data.columns.values:
        if column != 't':
            x = data[column].values  # returns a numpy array
            normalized_x = (x - x.min()) / (x.max() - x.min())


            data[column] = normalized_x
    return data


