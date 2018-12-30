import os
import cv2
import numpy as np
import pandas as pd
import landmarks as lm

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
def noise_accuracy(indx, noise_labels):
    f_neg = []
    for i in range(len(indx)):
        checker = np.sum(noise_labels[indx[i]])
        # iff all 4 labels are -1, img is noise
        if checker > -4:
            f_neg.append(indx[i])

    return f_neg


# Report detector noise performance
def report_noise(detector, time, noise, img_labels, img_paths):
    noise_count = len(noise)

    # determine accuracy of noisy images
    false_neg = noise_accuracy(noise, img_labels)
    accuracy = 1 - len(false_neg) / len(img_paths)

    f = open(os.path.join('out', detector, 'noise'), "w+")
    f.write("%d noisy images were detected in %0.2f min:\r\n"
            "Accuracy = %0.2f with grayscale and gamma correction\r\n\n"
            % (noise_count, time / 60, accuracy))
    [f.write("%s, " % img_num) for img_num in noise]
    f.write("\r\n\n\n%d false negatives were found (%0.2f FNR):\r\n\n"
            % (len(false_neg), len(false_neg) / noise_count))
    [f.write("%s, " % FN) for FN in false_neg]
    f.close()

    return


# Update + store features and labels
def update_features(detector):
    features, labels = lm.label_features(detector)

    # reshape features to 2D matrix
    m = features.shape
    features = np.reshape(features, (m[0], m[1] * 2)).astype(float)

    # convert to dataframe + store in csv
    df = pd.DataFrame(features)
    df.to_csv(os.path.join('out', detector, 'features.csv'))

    dl = pd.DataFrame(labels)
    dl.to_csv(os.path.join('out', detector, 'labels.csv'))

    return features, labels


