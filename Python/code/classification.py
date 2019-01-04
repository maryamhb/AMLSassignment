import os
import time
import utils as ut
import numpy as np
import pandas as pd
from sklearn import svm, neural_network as nn, preprocessing as pp
from sklearn.metrics import classification_report, confusion_matrix


# Labels: hair colour (0), glasses (1), smiling (2), young (3), human (4)
# Tasks:  emotion (1), age (2), glasses (3), human (4), hair colour (5)

def get_data(detector):
    #ut.update_features(detector)
    i_df = pd.read_csv(os.path.join('out', 'Face_detection', detector, 'names.csv'), usecols=[1])
    x_df = pd.read_csv(os.path.join('out', 'Face_detection', detector, 'features.csv'), usecols=[*range(1, 137)])
    y_df = pd.read_csv(os.path.join('out', 'Face_detection', detector, 'labels.csv'), usecols=[*range(1, 6)])
    # convert to np array
    i = i_df.values
    x = x_df.values
    y = y_df.values

    return i, x, y


def split_data(i, x, y):
    # 80% training data
    cut = int(len(i)*0.8)
    # Split training/test data
    training_x = x[:cut]
    training_y = y[:cut]
    test_i = i[cut:]
    test_x = x[cut:]
    test_y = y[cut:]

    return training_x, training_y, test_i, test_x, test_y


def train_svm(tr_img, tr_lb, te_img, te_lb, te_i, mod, task):

    # Initialise model
    model = svm.SVC(C=1, kernel='rbf', gamma='scale')

    # Fit model to data
    model.fit(tr_img, tr_lb)
    print("SVM Model:\n", model)
    # Predict test data
    prediction = model.predict(te_img)
    score = model.score(te_img, te_lb)
    print("\nScore: %0.5f" % score)

    conf_m = confusion_matrix(te_lb, prediction)

    # Report perf in csv
    ut.report_pred(mod, task, te_i, prediction, conf_m)

    return model


# Testing
te_model = 'Test'

all_i, all_x, all_y = get_data('Test')
tr_x, tr_y, te_nom, te_x, te_y = split_data(all_i, all_x, all_y)


# Map columns to tasks
tasks = {1: 3, 2: 1, 3: 2, 4: 4}

init = time.time()
for col in range(1, 5):
    print('Task', tasks[col], '-')
    train_svm(tr_x, tr_y[:, col], te_x, te_y[:, col], te_nom, te_model, tasks[col])

end = time.time()
ut.report_time(te_model, end-init)

print(end-init)




