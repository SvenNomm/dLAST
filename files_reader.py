#this function warps data reading from the file, calling processing functions and returns arrays of feature values for further processing

import os
import data_reader
import list_of_tests
import featureset_wrapper
import pandas as pd


def data_loader(PATH, depict, export, units, number_of_intervals, test_name):
    subjects_list = os.listdir(PATH)
    single_interval_masses = {}
    cumulative_interval_masses = {}

    for i in range(0, number_of_intervals):
        single_interval_masses[i] = pd.DataFrame()
        cumulative_interval_masses[i] = pd.DataFrame()

    for subject in subjects_list:
        print(subject)

        if ('KT' in subject) or ('PD' in subject):

            path_to_subject = PATH + subject + '/'
            test_files = os.listdir(path_to_subject)

            for test_file in test_files:
                if test_name in test_file:
                    # read the data from this file
                    file_name = path_to_subject + test_file
                    print('reading data from ', file_name)
                    print('File size is', os.path.getsize(file_name))

                    # compute features and add to the corresponding vector

                    test_data = data_reader.data_reader(file_name)
                    motion_masses_single_interval, motion_masses_cumulative_interval = featureset_wrapper.featureset_wrapper(test_data, number_of_intervals, units)

                    # append motion massess to the corresponding structures
                    for i in range(0, number_of_intervals):
                        #print(motion_masses_single_interval[i])
                        #print((single_interval_masses[i]).shape)
                        #print(motion_masses_single_interval[i].shape)

                        if motion_masses_single_interval[i].isnull().values.all():
                            print('single interval ', i, ' gives empty value')
                        else:
                            single_interval_masses[i] = single_interval_masses[i].append(motion_masses_single_interval[i], ignore_index=True)

                        if motion_masses_cumulative_interval[i].isnull().values.all():
                            print('cumulative interval ', i, ' gives empty values')
                        else:
                            cumulative_interval_masses[i] = cumulative_interval_masses[i].append(motion_masses_cumulative_interval[i], ignore_index=True)


    return single_interval_masses, cumulative_interval_masses