import numpy as np
import json
import matplotlib.pyplot as plt
import pandas as pd
import os
from matplotlib import rc
import pickle
import list_of_tests
import files_reader
import filter_models_wrapper
import ml_modeling_wrapper

from goodness_plot import *

test_names = list_of_tests.test_names()
interval_types = ['single', 'cumulative']

for test_nr in range(0, 6):
    #test_nr = 2

    #feature_set = []

    list_of_classifiers = ['kNN', 'DT', 'LR', 'SVM', 'RF']


    for interval_type in range(0,2):
        #interval_type = 0

        feature_set = ['A_m', 'D_m', 'P_m']

        #goodness_measures_list = ['precision', 'recall', 'recall',  'accuracy']
        goodness_measures_list = ['accuracy','specificity','sensitivity','precision']

        #PATH = '/Users/sven/kohalikTree/Data/MeDiag/DATA/interval_analysis/'
        PATH = 'C:/Users/Sven/Puu/Data_files/dLAST/interval_analysis/interval_analysis/'

        fname = PATH + test_names[test_nr] + '_PD_' + interval_types[interval_type] + '_interval_analysis,pkl'

        with open(fname, 'rb') as f:
            single_feature_values_PD = pickle.load(f)

        fname = PATH + test_names[test_nr] + '_KT_' + interval_types[interval_type] + '_interval_analysis,pkl'

        with open(fname, 'rb') as f:
            single_feature_values_KT = pickle.load(f)

        print('All the necessary files have been loaded.')

        #fishers_scores = filter_models_wrapper.filter_models_wrapper(single_feature_values_PD, single_feature_values_KT)
        goodness_measures = ml_modeling_wrapper.ml_models_wrapper_balanced(single_feature_values_PD, single_feature_values_KT,
                                                          feature_set, list_of_classifiers, goodness_measures_list)
        PATH = PATH + 'NG_results/'
        goodness_plot(goodness_measures, goodness_measures_list, test_names, test_nr, interval_types,interval_type, list_of_classifiers, PATH)

print('Thants all folks!')


