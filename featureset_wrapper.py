# this function wrapps the feature engeneering process and returns the vector of feture values
import list_of_tests
import noise_filer
import feature_constructor
import pandas as pd


def featureset_wrapper(dataset, number_of_intervals, units):
    # based on the screen resolution perform test segmentation
    screen_x_max, screen_y_max = list_of_tests.screen_resolution()
    interval_length = screen_x_max / number_of_intervals

    single_interval_masses = {}
    cumulative_interval_masses = {}

    for i in range(0, number_of_intervals):
        single_interval_masses[i] = pd.DataFrame()
        cumulative_interval_masses[i] = pd.DataFrame()

    for i in range(0, number_of_intervals):
        interval_min = i * interval_length
        interval_max = interval_min + interval_length
        interval_data_cumulative = dataset.loc[dataset['x'] < interval_max]
        interval_data_single = interval_data_cumulative.loc[interval_data_cumulative['x'] > interval_min]

        interval_data_cumulative = noise_filer.noise_filter(interval_data_cumulative)
        interval_data_single = noise_filer.noise_filter(interval_data_single)
        motion_mass = feature_constructor.feature_constructor(interval_data_single, units)
        single_interval_masses[i] = single_interval_masses[i].append(motion_mass)

        motion_mass = feature_constructor.feature_constructor(interval_data_cumulative, units)
        cumulative_interval_masses[i] = cumulative_interval_masses[i].append(motion_mass)

    return single_interval_masses, cumulative_interval_masses
