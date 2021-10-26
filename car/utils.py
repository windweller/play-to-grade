import numpy as np

def fill_na_with_mean(array):
    assert len(array.shape) == 2
    a = array
    col_mean = np.nanmean(a, axis=0)
    # Find indices that you need to replace
    inds = np.where(np.isnan(a))

    # Place column means in the indices. Align the arrays using take
    a[inds] = np.take(col_mean, inds[1])

    return a

def flatten(regular_list):
    flat_list = [item for sublist in regular_list for item in sublist]
    return flat_list
