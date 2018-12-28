import os
import numpy as np
import pandas as pd
import landmarks as lm
from sklearn import svm, neural_network as nn, preprocessing as pp
from sklearn.metrics import classification_report, confusion_matrix

# Labels: index (0), hair colour (1), glasses (2), smiling (3), young (4), human (5)

def get_data():
    # lm.update_features()
    x_df = pd.read_csv(os.path.join('out', 'features.csv'))
    y_df = pd.read_csv(os.path.join('out', 'labels.csv'))
    # convert to np array
    x = x_df.values
    y = y_df.values

    return x, y


def split_data(x, y):
    # Split training/test data
    training_x = x[:4000]
    training_y = y[:4000]
    test_x = x[4000:]
    test_y = y[4000:]

    return training_x, training_y, test_x, test_y


def train_svm(tr_img, tr_lb, te_img, te_lb):

    # Initialise model
    model = svm.SVC(C=1, kernel='linear', gamma='scale')

    # Fit model to data
    model.fit(tr_img, tr_lb[:, 2])
    print("SVM Model:\n", model)
    # Predict test data
    prediction = model.predict(te_img)
    score = model.score(te_img, te_lb[:, 2])
    print("Pred: ", prediction, "\nScore: %0.5f" % score)

    # Confusion matrices
    # print("Confusion Matrix:\n", confusion_matrix(te_lb.ravel(), prediction),
    #       "\nClassification Report:\n", classification_report(te_lb.ravel(), prediction))

    return model


# Testing
all_x, all_y = get_data()
tr_x, tr_y, te_x, te_y = split_data(all_x, all_y)


SVM_mod = train_svm(tr_x, tr_y, te_x, te_y)  # score = 0.84086 (after a long, long time)

