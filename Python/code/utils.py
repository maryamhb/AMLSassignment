import os
import cv2
import dlib
import time
import numpy as np
import pandas as pd
import landmarks as lm

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


# Report model predictions
def report_pred(model, t_num, names
                , predictions, conf_m):
    file = 'task_' + str(t_num) + '.csv'
    path = os.path.join('out', 'Classification', model)

    TN = conf_m[0, 0]
    FN = conf_m[0, 1]
    FP = conf_m[1, 0]
    TP = conf_m[1, 1]

    TPR = safe_div(TP, (TP+FN))
    TNR = safe_div(TN, (TN+FP))
    prec = safe_div(TP, (TP+FP))

    accuracy = (TN + TP)/len(names)
    print(accuracy)

    f = open(os.path.join(path, file), "w+")
    # Average inference accuracy
    f.write("%0.3f \r\n" % accuracy)
    # Predictions
    [f.write("%s, %d\r\n" % (str(int(names[i]))+'.png', predictions[i])) for i in range(len(names))]
    f.close()

    # Report performance
    if t_num == 5:
        f = open(os.path.join(path, 'performance.csv'), "w+")
        f.write("Task, Accuracy, TP, TN, FP, FN, TPR, TNR, Precision \r\n")

    else:
        f = open(os.path.join(path, 'performance.csv'), "a")

    f.write("%d, %0.3f, %d, %d, %d, %d, %0.3f, %0.3f, %0.3f \r\n"
            % (t_num, accuracy, TP, TN, FP, FN, TPR, TNR, prec))
    f.close()

    return


# Print time at the end of perf
def report_time(model, t):
    path = os.path.join('out', 'Classification', model)
    f = open(os.path.join(path, 'performance.csv'), "a")
    f.write("\nTime, %0.3f\r\n" % (int(t) / 60))

    return


