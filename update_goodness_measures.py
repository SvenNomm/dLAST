#

import pandas as pd
import numpy as np


def update_goodness_measures(goodness_measures_c, goodness_measures_s,  test_nr, list_of_classifiers):
    if test_nr == 0 or test_nr == 3:
        int_nr = 3
    else:
        int_nr = 0

    for k in range(0, 5):
        goodness_measures_c[list_of_classifiers[k]][:, int_nr] = goodness_measures_s[list_of_classifiers[k]][:, int_nr]

    return goodness_measures_c
