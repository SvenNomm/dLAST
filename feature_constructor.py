# this function wraps feature engeneering processa and returns the values of each element in the feature set
import convertors
import list_of_tests
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import euclidean_distances
import math


def feature_constructor(dataset, units):

    rows,cols = dataset.shape
    motion_mass_list = ['L_m', 'V_m', 'A_m', 'J_m', 'P_m', 'D_m', 'V_mean', 'A_mean', 'J_mean', 'P_mean', 'D_mean', 't_m']
    motion_mass_notations = ['$L_M$','$V_M$', '$A_M$', '$J_M$', '$P_M$', '$D_M$', 'mean V', 'mean A', 'mean J', 'mean P', 'mean D', 't']
    motion_mass = pd.DataFrame(columns=motion_mass_list, index=['0'])
    if rows > 3:
        dataset = dataset.reset_index(drop=True)
        screen_x_max, screen_y_max = list_of_tests.screen_resolution()
        x_mm2pix, y_mm2pix = list_of_tests.pixel_properties()

        dataset = convertors.convert_to_descartes_coordinates_from_iPad(dataset,screen_y_max)

        if units == 'mm':
            dataset = convertors.iPad2mm(dataset, x_mm2pix, y_mm2pix)

        rows, cols = dataset.shape
        loc_dist = np.zeros((rows, 1))
        velocity = np.zeros((rows, 1))
        acceleration = np.zeros((rows, 1))
        jerk = np.zeros((rows, 1))
        pressure_change = np.zeros((rows, 1))
        direction = np.zeros((rows, 1))
        direction_change = np.zeros((rows, 1))

        for i in range(1, rows-1):
            #print(i, dataset['x'][i],dataset['x'][i-1],dataset['y'][i],dataset['y'][i-1])
            loc_dist[i, 0] = np.sqrt((dataset['x'][i] - dataset['x'][i-1])**2 + (dataset['y'][i] - dataset['y'][i-1])**2)
            time_diff = dataset['t'][i] - dataset['t'][i-1]

            y_diff = dataset['y'][i] - dataset['y'][i-1]
            x_diff = dataset['x'][i] - dataset['x'][i-1]

            if (x_diff == 0) and (y_diff > 0):
                direction[i, 0] = 90

            if (x_diff == 0) and (y_diff < 0):
                direction[i, 0] = 270
            else:
                #print(x_diff)
                direction[i, 0] = math.degrees(np.arctan(y_diff / x_diff))

            direction_change[i, 0] = direction[i, 0] - direction[i-1, 0]
            velocity[i, 0] = loc_dist[i, 0] / time_diff
            acceleration[i, 0] = (velocity[i,0] - velocity[i-1, 0]) / time_diff
            jerk[i, 0] = (acceleration[i,0] - acceleration[i-1, 0]) / time_diff
            pressure_change[i, 0] = (dataset['p'][i] - dataset['p'][i-1])



        motion_mass['L_m'] = np.nansum(abs(loc_dist))
        motion_mass['t_m'] = dataset['t'][rows-1] - dataset['t'][0]
        motion_mass['V_m'] = np.nansum(abs(velocity))
        motion_mass['A_m'] = np.nansum(abs(acceleration))
        motion_mass['J_m'] = np.nansum(abs(jerk))
        motion_mass['P_m'] = np.nansum(abs(pressure_change))
        motion_mass['D_m'] = np.nansum(abs(direction_change))


        motion_mass['V_mean'] = np.nanmean(velocity)
        motion_mass['A_mean'] = np.nanmean(acceleration)
        motion_mass['J_mean'] = np.nanmean(jerk)
        motion_mass['P_mean'] = np.nanmean(pressure_change)
        motion_mass['D_mean'] = np.nanmean(direction_change)


    return motion_mass
