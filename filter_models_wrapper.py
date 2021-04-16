# this function returns the vectors of scores describing discriminating power of the features
import pandas as pd
import numpy as np
from fishers_score_pd import fishers_score

def filter_models_wrapper(feature_values_PD, feature_values_KT):
    columns = feature_values_PD[0].head()

    number_of_intervals = len(feature_values_KT)

    fishers_scores = pd.DataFrame()

    for i in range(0, number_of_intervals):
        df_length_PD = feature_values_PD[i].shape[0]
        df_length_KT = feature_values_KT[i].shape[0]
        if (df_length_KT > 0) and (df_length_PD > 0):
            labels_PD = []
            for r in range(0, df_length_PD):
                labels_PD.append(1)

            labels_KT = []
            for r in range(0, df_length_KT):
                labels_KT.append(0)

            # merge dataframes and lables
            a = feature_values_PD[i]
            feature_values = pd.concat([feature_values_PD[i], feature_values_KT[i]])
            labels = np.array(labels_PD + labels_KT)
            f_s = fishers_score(feature_values, labels)
            fishers_scores = fishers_scores.append(f_s, ignore_index=True)



    return fishers_scores



