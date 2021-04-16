# this function wrapps the ML modeling

import pandas as pd
import numpy as np
from fishers_score_pd import fishers_score
import sklearn
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier as knn
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from classifiers_wrapper import classifiers_wrapper
from sklearn.model_selection import train_test_split
from random import randint


def ml_models_wrapper(feature_values_PD, feature_values_KT, feature_set, list_of_classifiers):
    columns = feature_values_PD[4].columns.values
    motion_mass_notations = ['$L_M$', '$V_M$', '$A_M$', '$J_M$', '$P_M$', '$D_M$', 'mean V', 'mean A', 'mean J',
                             'mean P',
                             'mean D', 't']

    motion_mass_notations = pd.DataFrame([motion_mass_notations], columns=columns)

    number_of_intervals = len(feature_values_KT)

    # intialize goodness measures
    goodness_measures = {}
    for k in list_of_classifiers:
        goodness_measures[k] = np.zeros((4, number_of_intervals))

    for i in range(0, number_of_intervals):
        print('interval:', i)
        df_length_PD = feature_values_PD[i].shape[0]
        df_length_KT = feature_values_KT[i].shape[0]

        if (df_length_KT > 5) and (df_length_PD > 5):
            labels_PD = []
            for r in range(0, df_length_PD):
                labels_PD.append(1)

            labels_KT = []
            for r in range(0, df_length_KT):
                labels_KT.append(0)

            # merge dataframes and lables
            feature_values = pd.concat([feature_values_PD[i], feature_values_KT[i]])
            print('sample size:', feature_values.shape[0])
            labels = np.array(labels_PD + labels_KT)
            f_s = fishers_score(feature_values, labels)

            columns = f_s.columns.values

            one_row = np.asarray(f_s)
            cols = one_row.shape[1]
            new_columns = columns[one_row.argsort()[0, cols - 3:cols]]
            new_columns = list(new_columns)
            # new_columns = feature_set
            # evaluating differnt classifiers in a little bit dumb manner

            clf = knn(n_neighbors=5)
            print(list_of_classifiers[0])

            goodness_measures[list_of_classifiers[0]][:, i] = classifiers_wrapper(feature_values[new_columns], labels,
                                                                                  clf)[:, 0]

            clf = tree.DecisionTreeClassifier()
            print(list_of_classifiers[1])
            goodness_measures[list_of_classifiers[1]][:, i] = classifiers_wrapper(feature_values[new_columns], labels,
                                                                                  clf)[:, 0]

            clf = LogisticRegression(random_state=0)
            print(list_of_classifiers[2])
            goodness_measures[list_of_classifiers[2]][:, i] = classifiers_wrapper(feature_values[new_columns], labels,
                                                                                  clf)[:, 0]

            clf = svm.SVC()
            print(list_of_classifiers[3])
            goodness_measures[list_of_classifiers[3]][:, i] = classifiers_wrapper(feature_values[new_columns], labels,
                                                                                  clf)[:, 0]

            clf = RandomForestClassifier(max_depth=2, random_state=0)
            print(list_of_classifiers[4])
            goodness_measures[list_of_classifiers[4]][:, i] = classifiers_wrapper(feature_values[new_columns], labels,
                                                                                  clf)[:, 0]

            # fig = plt.figure()
            # ax = fig.add_subplot(111, projection='3d')

            # pd_values = feature_values_PD[i]
            # kt_values = feature_values_KT[i]

            # ax.scatter(pd_values[new_columns[0]], pd_values[new_columns[1]], pd_values[new_columns[2]],
            #          edgecolors='gold', facecolor='gold')

            # ax.scatter(kt_values[new_columns[0]], kt_values[new_columns[1]], kt_values[new_columns[2]],
            #         edgecolors='blue', facecolor='blue')

            # ax.set_xlabel(motion_mass_notations[new_columns[0]][0])
            # ax.set_ylabel(motion_mass_notations[new_columns[1]][0])
            # ax.set_zlabel(motion_mass_notations[new_columns[2]][0])
            # plt.show()
            # fishers_scores = fishers_scores.append(f_s)
            print('Hello!')
    return goodness_measures


def ml_models_wrapper_balanced(feature_values_PD, feature_values_KT, feature_set, list_of_classifiers):
    columns = feature_values_PD[4].columns.values
    motion_mass_notations = ['$L_M$', '$V_M$', '$A_M$', '$J_M$', '$P_M$', '$D_M$', 'mean V', 'mean A', 'mean J',
                             'mean P',
                             'mean D', 't']

    motion_mass_notations = pd.DataFrame([motion_mass_notations], columns=columns)

    number_of_intervals = len(feature_values_KT)

    # intialize goodness measures
    goodness_measures = {}
    for k in list_of_classifiers:
        goodness_measures[k] = np.zeros((4, number_of_intervals))

    for i in range(0, number_of_intervals):
        print('interval:', i)
        df_length_PD = feature_values_PD[i].shape[0]
        df_length_KT = feature_values_KT[i].shape[0]

        if (df_length_KT > 5) and (df_length_PD > 5):

            # ballance the data set

            if df_length_KT > df_length_PD:
                prop = df_length_PD / df_length_KT
                print(prop)
                feature_values_KT_balanced, left_over = train_test_split(feature_values_KT[i], train_size=prop)
                feature_values_PD_balanced = feature_values_PD[i]
            else:
                prop = df_length_KT / df_length_PD
                feature_values_PD_balanced, left_over = train_test_split(feature_values_PD[i], train_size=prop)
                feature_values_KT_balanced = feature_values_KT[i]

            df_length_PD = feature_values_PD_balanced.shape[0]
            df_length_KT = feature_values_KT_balanced.shape[0]

            labels_PD = []
            for r in range(0, df_length_PD):
                labels_PD.append(1)

            labels_KT = []
            for r in range(0, df_length_KT):
                labels_KT.append(0)

            # merge dataframes and lables
            feature_values = pd.concat([feature_values_PD_balanced, feature_values_KT_balanced])
            print('sample size:', feature_values.shape[0])
            labels = np.array(labels_PD + labels_KT)
            f_s = fishers_score(feature_values, labels)

            columns = f_s.columns.values

            one_row = np.asarray(f_s)
            cols = one_row.shape[1]
            new_columns = columns[one_row.argsort()[0, cols - 3:cols]]
            new_columns = list(new_columns)
            # new_columns = feature_set
            # evaluating differnt classifiers in a little bit dumb manner

            clf = knn(n_neighbors=5)
            print(list_of_classifiers[0])

            goodness_measures[list_of_classifiers[0]][:, i] = classifiers_wrapper(feature_values[new_columns], labels,
                                                                                  clf)[:, 0]

            clf = tree.DecisionTreeClassifier()
            print(list_of_classifiers[1])
            goodness_measures[list_of_classifiers[1]][:, i] = classifiers_wrapper(feature_values[new_columns], labels,
                                                                                  clf)[:, 0]

            clf = LogisticRegression(random_state=0)
            print(list_of_classifiers[2])
            goodness_measures[list_of_classifiers[2]][:, i] = classifiers_wrapper(feature_values[new_columns], labels,
                                                                                  clf)[:, 0]

            clf = svm.SVC()
            print(list_of_classifiers[3])
            goodness_measures[list_of_classifiers[3]][:, i] = classifiers_wrapper(feature_values[new_columns], labels,
                                                                                  clf)[:, 0]

            clf = RandomForestClassifier(max_depth=2, random_state=0)
            print(list_of_classifiers[4])
            goodness_measures[list_of_classifiers[4]][:, i] = classifiers_wrapper(feature_values[new_columns], labels,
                                                                                  clf)[:, 0]

            # fig = plt.figure()
            # ax = fig.add_subplot(111, projection='3d')

            # pd_values = feature_values_PD[i]
            # kt_values = feature_values_KT[i]

            # ax.scatter(pd_values[new_columns[0]], pd_values[new_columns[1]], pd_values[new_columns[2]],
            #          edgecolors='gold', facecolor='gold')

            # ax.scatter(kt_values[new_columns[0]], kt_values[new_columns[1]], kt_values[new_columns[2]],
            #         edgecolors='blue', facecolor='blue')

            # ax.set_xlabel(motion_mass_notations[new_columns[0]][0])
            # ax.set_ylabel(motion_mass_notations[new_columns[1]][0])
            # ax.set_zlabel(motion_mass_notations[new_columns[2]][0])
            # plt.show()
            # fishers_scores = fishers_scores.append(f_s)
            print('Hello!')
    return goodness_measures


def ml_models_wrapper_balanced_2(feature_values_PD, feature_values_KT, feature_set, list_of_classifiers):
    columns = feature_values_PD[4].columns.values
    motion_mass_notations = ['$L_M$', '$V_M$', '$A_M$', '$J_M$', '$P_M$', '$D_M$', 'mean V', 'mean A', 'mean J',
                             'mean P',
                             'mean D', 't']

    motion_mass_notations = pd.DataFrame([motion_mass_notations], columns=columns)

    number_of_intervals = len(feature_values_KT)

    # intialize goodness measures
    goodness_measures = {}
    for k in list_of_classifiers:
        goodness_measures[k] = np.zeros((4, number_of_intervals))

    row_numbers = []
    i = 0
    while i < 17:
        rn = randint(0, 17)
        if rn not in row_numbers:
            row_numbers.append(rn)
            i = i + 1

    row_numbers = np.asarray(row_numbers)
    print(row_numbers)

    for i in range(0, number_of_intervals):
        print('interval:', i)
        df_length_PD = feature_values_PD[i].shape[0]
        df_length_KT = feature_values_KT[i].shape[0]

        if (df_length_KT > 5) and (df_length_PD > 5):
            feature_values_KT_balanced = feature_values_KT[i].iloc[row_numbers]
            feature_values_PD_balanced = feature_values_PD[i]

            df_length_PD = feature_values_PD_balanced.shape[0]
            df_length_KT = feature_values_KT_balanced.shape[0]

            labels_PD = []
            for r in range(0, df_length_PD):
                labels_PD.append(1)
                labels_KT = []

            for r in range(0, df_length_KT):
                labels_KT.append(0)

            # merge dataframes and lables
            feature_values = pd.concat([feature_values_PD_balanced, feature_values_KT_balanced])
            print('sample size:', feature_values.shape[0])
            labels = np.array(labels_PD + labels_KT)
            f_s = fishers_score(feature_values, labels)

            columns = f_s.columns.values

            one_row = np.asarray(f_s)
            cols = one_row.shape[1]
            new_columns = columns[one_row.argsort()[0, cols - 3:cols]]
            new_columns = list(new_columns)
            # new_columns = feature_set
            # evaluating differnt classifiers in a little bit dumb manner

            clf = knn(n_neighbors=5)
            print(list_of_classifiers[0])

            goodness_measures[list_of_classifiers[0]][:, i] = classifiers_wrapper(feature_values[new_columns], labels,
                                                                              clf)[:, 0]

            clf = tree.DecisionTreeClassifier()
            print(list_of_classifiers[1])
            goodness_measures[list_of_classifiers[1]][:, i] = classifiers_wrapper(feature_values[new_columns], labels,
                                                                              clf)[:, 0]

            clf = LogisticRegression(random_state=0)
            print(list_of_classifiers[2])
            goodness_measures[list_of_classifiers[2]][:, i] = classifiers_wrapper(feature_values[new_columns], labels,
                                                                              clf)[:, 0]

            clf = svm.SVC()
            print(list_of_classifiers[3])
            goodness_measures[list_of_classifiers[3]][:, i] = classifiers_wrapper(feature_values[new_columns], labels,
                                                                              clf)[:, 0]

            clf = RandomForestClassifier(max_depth=2, random_state=0)
            print(list_of_classifiers[4])
            goodness_measures[list_of_classifiers[4]][:, i] = classifiers_wrapper(feature_values[new_columns], labels,
                                                                              clf)[:, 0]

            # fig = plt.figure()
            # ax = fig.add_subplot(111, projection='3d')

            # pd_values = feature_values_PD[i]
            # kt_values = feature_values_KT[i]

            # ax.scatter(pd_values[new_columns[0]], pd_values[new_columns[1]], pd_values[new_columns[2]],
            #          edgecolors='gold', facecolor='gold')

            # ax.scatter(kt_values[new_columns[0]], kt_values[new_columns[1]], kt_values[new_columns[2]],
            #         edgecolors='blue', facecolor='blue')

            # ax.set_xlabel(motion_mass_notations[new_columns[0]][0])
            # ax.set_ylabel(motion_mass_notations[new_columns[1]][0])
            # ax.set_zlabel(motion_mass_notations[new_columns[2]][0])
            # plt.show()
            # fishers_scores = fishers_scores.append(f_s)
            print('Hello!')
    return goodness_measures
