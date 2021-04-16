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

test_names = list_of_tests.test_names()
test_nr = 0
interval_type = 0

print(test_names[test_nr])

interval_types = ['single', 'cumulative']
print(interval_types[interval_type])


PATH = '/Users/sven/kohalikTree/Data/MeDiag/DATA/interval_analysis/'

fname = PATH + test_names[test_nr] + '_PD_' + interval_types[interval_type] + '_interval_analysis,pkl'

with open(fname, 'rb') as f:
    single_feature_values_PD = pickle.load(f)

fname = PATH + test_names[test_nr] + '_KT_' + interval_types[interval_type] + '_interval_analysis,pkl'

with open(fname, 'rb') as f:
    single_feature_values_KT = pickle.load(f)

fishers_scores = filter_models_wrapper.filter_models_wrapper(single_feature_values_PD, single_feature_values_KT)

actual_length = fishers_scores.shape[0]
if actual_length < 10:
    for i in range(0, 10-actual_length):
        fishers_scores.loc[-1] = [0,0,0,0,0,0,0,0,0,0,0,0]
        fishers_scores.index = fishers_scores.index + 1  # shifting index
        fishers_scores = fishers_scores.sort_index()



columns = fishers_scores.columns.values

mmn = ['$L_M$', '$V_M$', '$A_M$', '$J_M$', '$P_M$', '$D_M$', "$\\overline{V}$", '$\\overline{A}$', '$\\overline{J}$', '$\\overline{P}$','$\\overline{D}$', 't']
motion_mass_notations = pd.DataFrame([mmn], columns=columns, index=[0])

#motion_mass_notations = pd.DataFrame(motion_mass_notations, columns=columns)

best_features = pd.DataFrame()

for i in fishers_scores.index:
    one_row =np.asarray(fishers_scores.loc[i, :])
    #print(one_row)
    new_columns = columns[np.argsort(one_row)[-5:]]
    a= motion_mass_notations[new_columns].values
    a = pd.DataFrame(a)
    best_features = best_features.append(a)
    #print(np.argsort(one_row)[-5:])
    print(columns[np.argsort(one_row)[-5:]])

#best_features=pd.DataFrame(best_features)
#fishers_scores.to_csv(fname)
print(best_features.to_latex(escape=False))

#fig, axis = plt.subplots()



#motion_mass_notations = pd.DataFrame([motion_mass_notations], columns=columns)


#for column in fishers_scores.columns.values:
    #print(fishers_scores[column])
#    line = axis.plot(fishers_scores[column], label=motion_mass_notations[column][0])

#axis.legend()
#plt.show()


new_legend = motion_mass_notations[new_columns].values

fig1, axis  = plt.subplots()
temp_frame = pd.DataFrame(columns = new_columns)
ticks = ['1','2','3','4','5','6','7','8','9','10']
ticks_positions = [0,1,2,3,4,5,6,7,8,9]
for k in new_columns:
    tempa = fishers_scores[k].T
    temp_frame[k] = tempa

ax = plt.gca()
assert plt.gcf() is fig1 # succeeds
temp_frame.plot(kind='bar', alpha=0.9, width=0.8, edgecolor='black', linewidth=0.5, ax=ax)
ax.legend(new_legend[0])
assert plt.gcf() is fig1 # Succeeds now, too
plt.xticks(ticks_positions, ticks, rotation=0)

plt.show()

figure_name = PATH + 'five_best_fishers_scores_' + test_names[test_nr]+ '_' + interval_types[interval_type] + '_interval_analysis.pdf'
fig1.savefig(figure_name)

print('Thants all folks!')