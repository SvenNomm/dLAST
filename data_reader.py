# this function reads data from the given file and returns them in the form of pandas data frame
import os
import json
import pandas as pd


def data_reader(file_name):
    with open(file_name) as test_results_file:
        test_data = json.load(test_results_file)

    test_data_frame = pd.DataFrame()

    strokes_counter = 0
    for j in test_data['data']:
        stroke = pd.DataFrame(j)
        test_data_frame = test_data_frame.append(stroke)
        strokes_counter = strokes_counter + 1


    print('data shape:', test_data_frame.shape, 'strokes:', strokes_counter)
    return test_data_frame
