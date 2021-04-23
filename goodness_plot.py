# this function plots and saves model goodness parameters in the form of 3D bar plots

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
#fig = plt.figure()
#ax = fig.add_subplot(111, projection='3d')


def goodness_plot(goodness_measures, goodness_measures_list, test_names, test_nr, interval_types, interval_type,
                  list_of_classifiers, PATH):
    ticks_x = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10']
    ticks_y_l = ['0', '0.25', '0.5', '0.75', '0', '0.25', '0.5', '0.75', '0', '0.25', '0.5', '0.75', '0', '0.25', '0.5', '0.75']
    ticks_y_l_positions = [0, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2, 2.25, 2.5, 2.75, 3, 3.25, 3.5, 3.75]
    ticks_y_r = ['accuracy', 'precision', 'sensitivity', 'specificity']
    ticks_y_r = goodness_measures_list
    ticks_y_r_positions = [0.8, 1.8, 2.8, 3.8]

    x_positions = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]
    x_positions = np.asarray(x_positions)

    #ticks_y = goodness_measures_list
    y_positions = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    y_positions = np.asarray(y_positions)

    #x_positions, y_positions = np.meshgrid(x_positions, y_positions)

    #x_positions = x_positions.flatten()
    #y_positions = y_positions.flatten()
    #z_positions = np.zeros(y_positions.shape).flatten()
    #dx = np.zeros(y_positions.shape).flatten() + 0.8
    #dy = np.zeros(y_positions.shape).flatten()  #+ 0.03

    fig = plt.figure()
    axis = plt.gca()
    #ax = fig.add_subplot(111, projection='3d')
    axis.set_ylim(0, 4)

    axis.grid(axis='y', color='0.65', linestyle='--', linewidth=0.5)
    y_pos = y_positions
    #z_pos = z_positions
    list_of_colors = ['blue', 'orange', 'green', 'red', 'purple']
    classifier_nr = 0
    for k in list_of_classifiers:  # this loop goes through all the classifiers
        x_pos = x_positions + classifier_nr * 1.5

        #dz = np.zeros((4, 10))
        for i in range(0, 4):   # here we go through four goodness measures
            dz = goodness_measures[k][i, :]
            y_pos = y_positions + i

            #dz = dz.flatten()
            #ax.bar3d(x_pos, y_pos, z_pos, dx, dy, dz, alpha=0.3, color=list_of_colors[classifier_nr], edgecolor='black', linewidth=0.5)
            if i == 3:
                plt.bar(x_pos, dz, width=1.4, alpha=0.7, bottom=y_pos, align='center', data=None, edgecolor='black', linewidth=0.5, color=list_of_colors[classifier_nr], label=k)
            else:
                plt.bar(x_pos, dz, width=1.4, alpha=0.7, bottom=y_pos, align='center', data=None, edgecolor='black',
                        linewidth=0.5, color=list_of_colors[classifier_nr])

        classifier_nr = classifier_nr + 1
    assert plt.gcf() is fig  # Succeeds now, too
    #plt.legend(loc='upper center', ncol=5)
    plt.legend(bbox_to_anchor=(1, 1.1), ncol=5)

    plt.xticks(x_positions, ticks_x, rotation=0)

    #ax.yaxis.set_ticks_position('left')
    plt.yticks(ticks_y_l_positions, ticks_y_l, rotation=0)
    ax2 = axis.twinx()
    ax2.set_ylim(0, 4)
    plt.yticks(ticks_y_r_positions, ticks_y_r, rotation=90)
    axis.set_xlabel('Intervals')

    #ax2.yaxis.tick_right()
    #ax2.set_yticks(ticks_y_r_positions, ticks_y_r)
    figure_name = PATH + 'fixed_goodness_of_the_models_' + test_names[test_nr] + '_' + interval_types[
        interval_type] + '_interval_analysis.pdf'
    fig.savefig(figure_name)


    plt.show()


