import os
import time
import utils as ut
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn import svm, neural_network as nn, preprocessing as pp

# Labels: hair colour (0), glasses (1), smiling (2), young (3), human (4)
# Tasks:  emotion (1), age (2), glasses (3), human (4), hair colour (5)


# Retrieve features and labels from
# csv files and update if necessary
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


# Partition training and test
# data into 80-20% of sample size
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


# Scale landmarks with sklearn
# preprocessing fit to training data
def scale_dat(training_img, test_img):
    scaler = pp.StandardScaler()
    scaler.fit(training_img)
    tr_img = scaler.transform(training_img)
    te_img = scaler.transform(test_img)

    return tr_img, te_img


# Support vector machine classifier
# Returns model, predictions and confusion matrix
def train_svm(kern, tr_img, tr_lb, te_img, te_lb):

    # Initialise model
    model = svm.SVC(C=1, kernel=kern, gamma='scale')

    # Fit model to data
    model.fit(tr_img, tr_lb)
    # Predict test data
    prediction = model.predict(te_img)
    score = model.score(te_img, te_lb)
    print("\nScore: %0.5f" % score)

    conf_m = confusion_matrix(te_lb, prediction)

    return model, prediction, conf_m


# Multi-layer perceptron classifier
# Returns model, predictions and confusion matrix
def train_mlp(solver, tr_img, tr_lb, te_img, te_lb):

    # Initialise model
    model = nn.MLPClassifier(solver=solver, alpha=1e-5,
                             hidden_layer_sizes=(3, 2), random_state=1)

    # Scale (x,y) coordinates
    tr_img, te_img, = scale_dat(tr_img, te_img)
    # Fit model to data
    model.fit(tr_img, tr_lb)
    # Predict test data
    prediction = model.predict(te_img)
    score = model.score(te_img, te_lb)
    print("Score: %0.5f" % score)  # 0.8824 score

    conf_m = confusion_matrix(te_lb, prediction)

    return model, prediction, conf_m


# Gather cross validation scores
# for varying parameters
def cross_val(clf, xAll, yAll):
    scores = []
    if clf == "SVM":
        for arg in range(1, 10):
            cv_model = svm.SVC(C=arg / 10, kernel='sigmoid', gamma='scale')
            cv_score = cross_val_score(cv_model, xAll, yAll, cv=10, scoring='accuracy')
            scores.append(cv_score.mean())
    else:
        for arg in range(1, 10):
            cv_model = nn.MLPClassifier(solver='lbfgs', alpha=1e-5,
                                     hidden_layer_sizes=(arg, arg), random_state=1)
            cv_score = cross_val_score(cv_model, xAll, yAll, cv=10, scoring='accuracy')
            scores.append(cv_score.mean())
    print(np.around(scores, 4))

    return 0


# Get data, run classifiers and store
# results into csv here to avoid globals
def test_mod():
    # Initialise lists
    pred_list = []
    models_list = []
    conf_m_list = []
    cv_score_list = []

    # Define model
    te_model = 'MLP'
    arg = 'lbfgs'
    all_i, all_x, all_y = get_data('HoG')
    tr_x, tr_y, te_nom, te_x, te_y = split_data(all_i, all_x, all_y)

    # Map columns to tasks
    tasks = {0: 5, 1: 3, 2: 1, 3: 2, 4: 4}

    init = time.time()

    # Classify task by task
    for col in range(0, 1):
        t_num = tasks[col]
        print('\nTask', t_num, '-')

        # Train model classifier
        model, pred, c_matrix = train_svm(arg, tr_x, tr_y[:, col],
                                          te_x, te_y[:, col])

        # 10-fold cross validation
        print('Cross validating')
        # 10-fold cross validation score
        cv_score = cross_val_score(model, all_x, all_y[:, col], cv=10,
                                   scoring='accuracy')

        # Write to lists for later
        print('Storing results')
        # Report performance in csv
        models_list.append(model)
        conf_m_list.append(c_matrix)
        pred_list.append(pred)
        cv_score_list.append(cv_score)

    end = time.time()

    # Store results in .csv
    print('Reporting performance')
    for col in range(1, 5):
        print('\nTask', tasks[col])
        ut.report_pred(te_model, arg, tasks[col], te_nom,
                       pred_list[col-1], conf_m_list[col-1], cv_score_list[col-1])

    # Plot learning curves
    print('Plotting graphs')
    for col in range(1, 5):
        print('\nTask', tasks[col])
        fig = ut.plot_learning_curve(models_list[col-1], te_model + ' ' + arg + ' Learning curve',
                                     all_x, all_y[:, col], (0.4, 1.01), cv=10, n_jobs=4)
        fig.savefig(os.path.join('out', 'Graphs', 'T' + str(tasks[col]) + '_' + te_model + '_' + arg + '.png'))

    ut.report_time(4, te_model, arg, end - init)

    print(end - init)

    return 0


test_mod()
