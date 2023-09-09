import numpy as np
import pandas as pd


def make_matrix(x):
    x = np.asarray(x)
    shape = x.shape
    if len(shape) < 2:
        x = x.reshape(-1, 1)

    return x


def scale(x, center=True, normalize=True, axis=0):
    if center:
        x = x - np.nanmean(x, axis=axis)  # x.mean(axis=axis)
    if normalize:
        x = x / np.nanstd(x, axis=axis)  # x.std(axis=axis)
    return x


def col_means(x):
    y = np.mean(x, axis=0)
    return y


def majority_vote(x):
    x_shape = x.shape
    if len(x_shape) < 2:
        y = np.bincount(x).argmax()
    else:
        y = np.zeros(x_shape[1])
        for i in range(x_shape[1]):
            y[i] = majority_vote(x[:, i])

    return y


def get_x_vars(x):
    if isinstance(x, pd.DataFrame):
        x_vars = x.columns.values
    else:
        x_shape = x.shape
        if len(x_shape) < 2:
            n = 1
        else:
            n = x.shape[1]
        x_vars = np.arange(n)

    return x_vars
