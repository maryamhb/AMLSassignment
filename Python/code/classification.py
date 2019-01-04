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


def scale_dat(training_img, test_img):
    scaler = pp.StandardScaler()
    scaler.fit(training_img)
    tr_img = scaler.transform(training_img)
    te_img = scaler.transform(test_img)

    return tr_img, te_img


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


def train_mlp(tr_img, tr_lb, te_img, te_lb, te_i, mod, task):

    # Initialise model
    model = nn.MLPClassifier(solver='lbfgs', alpha=1e-5,
                             hidden_layer_sizes=(3, 2), random_state=1)

    # Scale (x,y) coordinates
    tr_img, te_img, = scale_dat(tr_img, te_img)
    # Fit model to data
    model.fit(tr_img, tr_lb)
    print("MLP Model:\n", model)
    # Predict test data
    prediction = model.predict(te_img)
    score = model.score(te_img, te_lb)
    print("\nScore: %0.5f" % score)  # 0.8824 score

    conf_m = confusion_matrix(te_lb, prediction)

    # Report perf in csv
    ut.report_pred(mod, task, te_i, prediction, conf_m)

    return model


# Testing
te_model = 'MLP'

all_i, all_x, all_y = get_data('HoG')
tr_x, tr_y, te_nom, te_x, te_y = split_data(all_i, all_x, all_y)


# Map columns to tasks
tasks = {0: 5, 1: 3, 2: 1, 3: 2, 4: 4}

init = time.time()
for col in range(5):
    print('Task', tasks[col], '-')
    #train_svm(tr_x, tr_y[:, col], te_x, te_y[:, col], te_nom, te_model, tasks[col])
    train_mlp(tr_x, tr_y[:, col], te_x, te_y[:, col], te_nom, te_model, tasks[col])

end = time.time()
ut.report_time(te_model, end-init)

print(end-init)




