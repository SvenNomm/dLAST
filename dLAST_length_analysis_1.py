import numpy as np
import json
import matplotlib.pyplot as plt
import pandas as pd
import os
from matplotlib import rc
import pickle
import list_of_tests
import files_reader

test_names = list_of_tests.test_names()
test_nr = 3

number_of_intervals = 10

PATH_PD = '/Users/sven/kohalikTree/Data/MeDiag/DATA/PD/'
PATH_KT = '/Users/sven/kohalikTree/Data/MeDiag/DATA/KT/'
PATH = '/Users/sven/kohalikTree/Data/MeDiag/DATA/interval_analysis/'

single_feature_values_KT, cumulative_feature_values_KT = files_reader.data_loader(PATH_KT, False, False, 'mm', number_of_intervals, test_names[test_nr])
single_feature_values_PD, cumulative_feature_values_PD = files_reader.data_loader(PATH_PD, False, False, 'mm', number_of_intervals, test_names[test_nr])

fname = PATH + test_names[test_nr] + '_PD_' + 'single_interval_analysis,pkl'
output = open(fname, 'wb')
pickle.dump(single_feature_values_PD, output)
output.close()

fname = PATH + test_names[test_nr] + '_PD_' +'cumulative_interval_analysis,pkl'
output = open(fname, 'wb')
pickle.dump(cumulative_feature_values_PD, output)
output.close()

fname = PATH + test_names[test_nr] + '_KT_' +'single_interval_analysis,pkl'
output = open(fname, 'wb')
pickle.dump(single_feature_values_KT, output)
output.close()

fname = PATH + test_names[test_nr] + '_KT_' +'cumulative_interval_analysis,pkl'
output = open(fname, 'wb')
pickle.dump(cumulative_feature_values_KT, output)
output.close()


print('Thats all folks!')

