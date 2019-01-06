from sklearn.model_selection import cross_val_score
from sklearn import svm, neural_network as nn
from sklearn.metrics import confusion_matrix
from sklearn import preprocessing as pp
import pandas as pd
import os


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
    print("\nSVM Score: %0.5f" % score)

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
    print("\nMLP Score: %0.5f" % score)

    conf_m = confusion_matrix(te_lb, prediction)

    return model, prediction, conf_m


# Gather cross validation scores
# for varying parameters
def cross_val(clf, xAll, yAll, fold, loop, cv_mod):
    scores = []
    if loop:
        if clf == "SVM":
            for arg in range(1, 10):
                cv_model = svm.SVC(C=arg / 10, kernel='sigmoid', gamma='scale')
                cv_score = cross_val_score(cv_model, xAll, yAll, cv=fold, scoring='accuracy')
                scores.append(cv_score.mean())
        elif clf == "MLP":
            for arg in range(1, 10):
                cv_model = nn.MLPClassifier(solver='lbfgs', alpha=1e-5,
                                            hidden_layer_sizes=(arg, arg), random_state=1)
                cv_score = cross_val_score(cv_model, xAll, yAll, cv=fold, scoring='accuracy')
                scores.append(cv_score.mean())

        return scores

    else:
        cv_score = cross_val_score(cv_mod, xAll, yAll, cv=fold,
                                   scoring='accuracy')
        return cv_score
