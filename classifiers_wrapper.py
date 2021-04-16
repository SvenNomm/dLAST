import pandas as pd
import numpy as np
from fishers_score_pd import fishers_score
import sklearn
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier as knn
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.metrics import make_scorer
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import accuracy_score


def classifiers_wrapper(data, labels, clf, goodness_measures_list):
    # goodness_measures_list = ['accuracy','specificity','sensitivity','precision']
    scores = np.zeros((4, 1))
    idx = 0
    for goodness_measure in goodness_measures_list:
        print('Computing ', goodness_measure)
        if goodness_measure == 'sensitivity':
            scorer = make_scorer(recall_score, pos_label=1)
        elif goodness_measure == 'specificity':
            scorer = make_scorer(recall_score, pos_label=0)
        else: #goodness_measure == 'accuracy':
            gm = eval(goodness_measure + '_score')
            scorer = make_scorer(gm)


        cvs = cross_val_score(clf, data, labels, cv=5, scoring=scorer)
        print(cvs)
        scores[idx, 0] = np.average(cvs)
        idx = idx + 1



    #scores_accuracy = cross_val_score(clf, data, labels, cv=3)
    #scores_precision = cross_val_score(clf, data, labels, cv=3, scoring='precision')
    #scores_recall = cross_val_score(clf, data, labels, cv=3, scoring='recall')
    #specificity = cross_val_score(clf, data, labels, cv=3, scoring='recall')

    #scores[0, 0] = np.average(scores_accuracy)
    #scores[1, 0] = np.average(scores_recall)
    #scores[2, 0] = np.average(scores_precision)
    #scores[3, 0] = np.average(specificity)

    return scores
