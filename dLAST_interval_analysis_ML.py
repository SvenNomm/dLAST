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
test_nr = 3

#feature_set = []

list_of_classifiers = ['kNN', 'DT', 'LR', 'SVM', 'RF']

interval_types = ['single', 'cumulative']

interval_type = 0

feature_set = ['A_m', 'D_m', 'P_m']

goodness_measures_list = ['accuracy', 'precision', 'recall', 'F1-score']

PATH = '/Users/sven/kohalikTree/Data/MeDiag/DATA/interval_analysis/'

fname = PATH + test_names[test_nr] + '_PD_' + interval_types[interval_type] + '_interval_analysis,pkl'

with open(fname, 'rb') as f:
    single_feature_values_PD = pickle.load(f)

fname = PATH + test_names[test_nr] + '_KT_' + interval_types[interval_type] + '_interval_analysis,pkl'

with open(fname, 'rb') as f:
    single_feature_values_KT = pickle.load(f)



#fishers_scores = filter_models_wrapper.filter_models_wrapper(single_feature_values_PD, single_feature_values_KT)
goodness_measures = ml_modeling_wrapper.ml_models_wrapper(single_feature_values_PD, single_feature_values_KT,
                                                          feature_set, list_of_classifiers)

goodness_plot(goodness_measures, goodness_measures_list, test_names, test_nr, interval_types,interval_type, list_of_classifiers)
ticks = ['1','2','3','4','5','6','7','8','9','10']
ticks_positions = [0,1,2,3,4,5,6,7,8,9]
for i in range(0, 4):

    fig1, axis  = plt.subplots()
    temp_frame = pd.DataFrame(columns = list_of_classifiers)

    for k in list_of_classifiers:
        tempa = goodness_measures[k][i, :].T
        temp_frame[k] = tempa

        #line = axis.plot(goodness_measures[k][i, :], label=k)
    ax = plt.gca()
    assert plt.gcf() is fig1 # succeeds
    temp_frame.plot(kind='bar', alpha=0.9, width=0.8, edgecolor='black', linewidth=0.5, ax=ax)
    #axis.legend()
    assert plt.gcf() is fig1  # Succeeds now, too
    plt.xticks(ticks_positions, ticks, rotation=0)
    ax.set_ylim([0, 1.1])
    plt.show()
    figure_name = PATH + goodness_measures_list[i] + '_' + test_names[test_nr] + '_' + interval_types[
        interval_type] + '_interval_analysis.pdf'
    #fig1.savefig(figure_name)

    # plotting in the form of bars



print('Thants all folks!')