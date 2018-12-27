import os
import numpy as np
import pandas as pd
import landmarks as lm
from sklearn import svm, neural_network as nn, preprocessing as pp


def get_data():
    # lm.update_features()
    x_df = pd.read_csv(os.path.join('out', 'features.csv'))
    y_df = pd.read_csv(os.path.join('out', 'labels.csv'))
    # convert to np array
    x = x_df.values
    y = y_df.values
    # Split training/test data
    tr_X = x[:4000]
    tr_Y = y[:4000]
    te_X = x[4000:]
    te_Y = y[4000:]

    return tr_X.shape, tr_Y.shape, te_X.shape, te_Y.shape


# Testing
tr_x, tr_y, te_x, te_y = get_data()
