from sklearn.model_selection import learning_curve
from keras.preprocessing import image
import matplotlib.pyplot as plt
import landmarks as lm
import pandas as pd
import numpy as np
import dlib
import cv2
import os


# Data directory
img_dir = os.path.join('..', 'dataset')
labels_dir = os.path.join('..', 'attribute_list.csv')
pred_dir = os.path.join('models', 'shape_predictor_68_face_landmarks.dat')
haar_dir = os.path.join('models', 'haarcascade_frontalface_default.xml')
model_dir = os.path.join('models', 'opencv_face_detector_uint8.pb')
config_dir = os.path.join('models', 'opencv_face_detector.pbtxt')
mmod_dir = os.path.join('models', 'mmod_human_face_detector.dat')


# Models and predictors
predictor = dlib.shape_predictor(pred_dir)
hog_detector = dlib.get_frontal_face_detector()
face_cascade = cv2.CascadeClassifier(haar_dir)
net = cv2.dnn.readNetFromTensorflow(model_dir, config_dir)
dnnFaceDetector = dlib.cnn_face_detection_model_v1(mmod_dir)


# Util functions

# Load images and labels
def load_data(col1, coln):
    # get directories
    img_paths = [os.path.join(img_dir, l) for l in os.listdir(img_dir)]
    # order items
    img_paths = sorted(sorted(img_paths), key=len, reverse=False)

    labels_file = open(labels_dir, 'r')
    lines = labels_file.readlines()
    # store all labels in a dictionary
    img_labels = {line.split(',')[0]: [int(line.split(',')[col]) for col in range(col1, coln)] for line in lines[2:]}

    return img_paths, img_labels


# Rect to bounding box
def rect2bb(rect):
    # convert dlib detector's rect to bounding box for convenience
    x = rect.left()
    y = rect.top()
    w = rect.right() - x
    h = rect.bottom() - y

    return x, y, w, h


# Convert landmarks to xy 2-tuple
def shape2np(shape, dtype="int"):
    # init list of x,y coordinates
    xy = np.zeros((shape.num_parts, 2), dtype=dtype)

    for i in range(shape.num_parts):
        xy[i] = (shape.part(i).x, shape.part(i).y)

    return xy


# Gamma correction
def correct_gamma(img, gamma):
    gamma_inv = 1.0 / gamma
    table = np.array([((i / 255.0) ** gamma_inv) * 255
                      for i in np.arange(0, 256)]).astype("uint8")

    # gamma correction using lookup table
    gam = cv2.LUT(img, table)

    return gam


# Test face detection accuracy
def noise_accuracy(noise, img_labels):
    f_neg = []
    f_pos = []

    for i in range(1, len(img_labels)):
        checker = np.sum(img_labels[str(i)])
        if str(i) in noise:
            if checker > -4:
                # iff all 5 labels are -1, img is noise
                f_neg.append(i)
        else:
            if checker < -4:
                f_pos.append(i)

    return f_neg, f_pos


# Report detector noise performance
def report_noise(det, time, noise, img_labels, img_paths):
    noise_count = len(noise)
    img_count = len(img_labels)

    # determine accuracy of noisy images
    false_neg, false_pos = noise_accuracy(noise, img_labels)
    FN_count = len(false_neg)
    FP_count = len(false_pos)
    true_noise = noise_count-FN_count
    true_face = img_count - true_noise

    accuracy = 1 - (len(false_neg)+len(false_pos)) / len(img_paths)

    f = open(os.path.join('out', 'Face_detection', det, 'noise'), "w+")
    f.write("%s: %d faces were detected in %0.2f min:\r\n"
            "%d noisy images with  %0.2f accuracy\r\n\n"
            % (det, img_count-noise_count, time / 60, noise_count, accuracy))
    [f.write("%s, " % img_num) for img_num in noise]
    f.write("\r\n\n\n%d false negatives were found (%0.2f FNR):\r\n\n"
            % (FN_count, FN_count / true_noise))
    [f.write("%s, " % FN) for FN in false_neg]
    f.write("\r\n\n\n%d false positives were found (%0.2f FPR):\r\n\n"
            % (FP_count, FP_count / true_face))
    [f.write("%s, " % FP) for FP in false_pos]
    f.close()

    return


# Update + store features and labels
def update_features(detector):
    index, features, labels = lm.label_features(detector)

    # reshape features to 2D matrix
    m = features.shape
    features = np.reshape(features, (m[0], m[1] * 2)).astype(float)

    # convert to dataframe + store in csv
    df = pd.DataFrame(index)
    df.to_csv(os.path.join('out', 'Face_detection', detector, 'names.csv'))

    df = pd.DataFrame(features)
    df.to_csv(os.path.join('out', 'Face_detection', detector, 'features.csv'))

    dl = pd.DataFrame(labels)
    dl.to_csv(os.path.join('out', 'Face_detection', detector, 'labels.csv'))

    return features, labels


# Return zero if denominator = 0
def safe_div(num, den):
    if den == 0:
        return 0
    else:
        return num/den


# Report binary model predictions into csv
def report_binary(model, arg, t_num, names,
                  predictions, conf_m, cv_score):
    file = 'task_' + str(t_num) + '.csv'
    path = os.path.join('out', 'Classification', model, arg)

    TN = conf_m[0, 0]
    FN = conf_m[0, 1]
    FP = conf_m[1, 0]
    TP = conf_m[1, 1]

    TPR = safe_div(TP, (TP+FN))
    TNR = safe_div(TN, (TN+FP))
    prec = safe_div(TP, (TP+FP))

    accuracy = (TN + TP)/len(names)

    f = open(os.path.join(path, file), "w+")
    # Average inference accuracy
    f.write("%0.3f \r\n" % accuracy)
    # Predictions
    [f.write("%s, %d\r\n" % (str(int(names[i]))+'.png', predictions[i])) for i in range(len(names))]
    f.close()

    # Report performance
    perf_file = 'binary'
    first_t = 3

    if t_num == first_t:
        f = open(os.path.join(path, perf_file + '.csv'), "w+")
        f.write("Task, Accuracy, TP, TN, FP, FN, TPR, TNR, Precision, CV-ave\r\n")

    else:
        f = open(os.path.join(path, perf_file + '.csv'), "a")

    f.write("%d, %0.3f, %d, %d, %d, %d, %0.3f, %0.3f, %0.3f, %0.3f\r\n"
            % (t_num, accuracy, TP, TN, FP, FN, TPR, TNR, prec, np.mean(cv_score)))
    f.close()

    return


# Report multiclass model predictions into csv
def report_multiclass(model, arg, names,
                      predictions, conf_m):
    # define path
    file = 'task_5.csv'
    path = os.path.join('out', 'Classification', model, arg)

    # calculate accuracy
    true_sum = 0
    for i in range(0,len(conf_m)):
        true_sum += conf_m[i, i]
    score = true_sum/np.sum(conf_m)

    f = open(os.path.join(path, file), "w+")
    # Average inference accuracy
    f.write("%0.3f \r\n" % score)
    # Predictions
    [f.write("%s, %d\r\n" % (str(int(names[i])) + '.png', predictions[i])) for i in range(len(names))]
    f.close()

    perf_file = os.path.join(path, 'multiclass.csv')

    df = pd.DataFrame(conf_m)
    df.to_csv(perf_file)

    f = open(perf_file, "a")
    # Average inference accuracy
    f.write("\r\nAccuracy,")
    # Predictions
    f.write("%0.2f\r\n" % score)
    f.close()

    return 0


# Print time at the end of perf
def report_time(t_num, model, arg, t):
    file = 'multiclass' if t_num == 5 else 'binary'
    path = os.path.join('out', 'Classification', model, arg)
    f = open(os.path.join(path, file+'.csv'), "a")
    f.write("\nTime, %0.3f\r\n" % (int(t) / 60))
    f.close()

    return


# Plot learning curve
def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):

    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt


# Remove noise with previously-run HoG data
def denoise_training(tr_i, tr_x, tr_y):
    tr_xx = []
    tr_yy = []
    # access filtered names
    i_df = pd.read_csv(os.path.join('out', 'Face_detection',
                                    'HoG', 'names.csv'), usecols=[1])
    for i in tr_i:
        if i in i_df:
            tr_xx.append(tr_x[i])
            tr_yy.append(tr_y[i])

    tr_xx = np.array(tr_x)
    tr_yy = np.array(tr_y).ravel()

    print(tr_xx.shape, tr_yy.shape)

    return tr_xx, tr_yy


# Remove noisy data by running HoG
def HoG_denoise_tr(img_paths, img_labels, tr_n):
    tr_i = []
    tr_x = []
    tr_y = []

    for img_path in img_paths:
        img_name = img_path.split('.')[2].split('/')[-1]

        img = image.img_to_array(image.load_img(img_path,
                                                target_size=None,
                                                interpolation='bicubic'))
        faces, _ = lm.get_landmarks('HoG', img)
        if faces is not None:
            # Halve image size x3 to reduce complexity
            img = cv2.resize(img, (32, 32))

            tr_i.append(img_name)
            tr_x.append(img)
            tr_y.append(img_labels[img_name])

        if img_path == img_paths[tr_n]: break

    tr_img = np.array(tr_x)
    tr_yy = np.array(tr_y).ravel()

    print(tr_img.shape, tr_yy.shape)

    # reshape images to 2D matrix
    m = tr_img.shape
    tr_xx = np.reshape(tr_img, (m[0], m[1] * m[2] * 3)).astype(float)

    print(tr_xx.shape, tr_yy.shape)

    return tr_xx, tr_yy


