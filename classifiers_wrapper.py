import pandas as pd
import numpy as np
from fishers_score_pd import fishers_score
import sklearn
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier as knn
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from sklearn import tree


def classifiers_wrapper(data, labels, clf):
    scores = np.zeros((4, 1))

    scores_accuracy = cross_val_score(clf, data, labels, cv=3)
    scores_precision = cross_val_score(clf, data, labels, cv=3, scoring='precision')
    scores_recall = cross_val_score(clf, data, labels, cv=3, scoring='recall')
    scores_f1 = cross_val_score(clf, data, labels, cv=3, scoring='f1')

    scores[0, 0] = np.average(scores_accuracy)
    scores[1, 0] = np.average(scores_recall)
    scores[2, 0] = np.average(scores_precision)
    scores[3, 0] = np.average(scores_f1)

    return scores
