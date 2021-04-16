import pandas as pd
import numpy as np
import convertors

def fishers_score(features_values, labels):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    #features_values = convertors.normalize(features_values)
    features_values = convertors.minmax_normalize(features_values)
    # if values.dtypes in numerics:
    n = len(features_values)
    class_names = np.unique(labels)
    number_of_classes, = class_names.shape

    columns = features_values.columns.values
    fishers_scores = pd.DataFrame(index=[0], columns=columns)

    for column in columns:
        num = 0
        denum = 0
        sample_mean = np.mean(features_values[column])
        for i in range(0, number_of_classes):
            class_values = features_values[column][labels == class_names[i,]]
            class_mean = np.mean(class_values)
            class_std = np.std(class_values)
            class_prop = len(class_values) / n

        num = num + class_prop * (sample_mean - class_mean) ** 2
        denum = denum + class_prop * (class_std) ** 2
            # print('class=', i, ' class_mean=', class_mean, 'class_p=', class_prop, 'class_std=', class_std)
            # print('num=', num, 'denum=', denum)

        if denum == 0:
            f_s = 0
        else:
            f_s = num / denum

        fishers_scores[column][0] = f_s

    return fishers_scores
