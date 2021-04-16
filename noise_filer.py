# this function eliminates noise points which time stamp is not consequent to the neighboring  points of the interval


def noise_filter(dataset):
    rows, cols = dataset.shape
    if rows > 3:
        for i in range(rows-1, 1):
            if (dataset[i,'t'] < dataset[i-1,'t']) or (dataset[i,'t'] > dataset[i+1,'t']):
                # exclude given row from the dataframe
                dataset.drop(dataset.index[[i]])
                print('row', i, 'deleted')

    return dataset
