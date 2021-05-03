# this function computes ranges for goodness prameters


import pandas as pd
import numpy as np


def compute_ranges(goodness_measures, list_of_classifiers, test_nr, goodness_measures_list):
    if test_nr == 0 or test_nr == 3:
        int_nr = 3
    else:
        int_nr = 0

    rows = len(list_of_classifiers)
    cols = len(goodness_measures_list)
    ranges = np.zeros((cols, rows))

    for i in range(0, cols): # to list all the classifiers
        for k in range(0, rows): # for each goodness measure
             ranges[i,k] = np.max(goodness_measures[list_of_classifiers[k]][i, int_nr:10]) \
                           - np.min(goodness_measures[list_of_classifiers[k]][i, int_nr:10])

    pandas_ranges = pd.DataFrame(ranges, columns=list_of_classifiers, index=goodness_measures_list)
    print(pandas_ranges.to_latex())
    return pandas_ranges
