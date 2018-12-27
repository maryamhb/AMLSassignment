import os
import numpy as np
import pandas as pd
from keras.preprocessing import image
import cv2
import dlib
import time

start = time.time()

# Data directory
img_dir = os.path.join('..', 'dataset')
labels_dir = os.path.join('..', 'attribute_list.csv')

# Face detector + landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')


# Run functions through dataset

def label_features():
    # load image + label data
    img_paths = [os.path.join(img_dir, l) for l in os.listdir(img_dir)]
    # order items
    img_paths = sorted(sorted(img_paths), key=len, reverse=False)

    labels_file = open(labels_dir, 'r')
    lines = labels_file.readlines()
    # Store all labels in a dictionary
    img_labels = {line.split(',')[0] : [int(line.split(',')[col]) for col in range(1, 6)] for line in lines[2:]}

    if os.path.isdir(img_dir):
        all_features = []
        all_labels = []
        noise = []
        data_count = 0
        noise_count = 0

        for img_path in img_paths:
            img_name = img_path.split('.')[2].split('/')[-1]

            img = image.img_to_array(image.load_img(img_path,
                                                    target_size=None,
                                                    interpolation='bicubic'))
            img_features, _ = get_landmarks(img)
            if img_features is not None:
                all_features.append(img_features)
                all_labels.append(img_labels[img_name])
                data_count += 1

            else:
                noise.append(img_name)
                noise_count += 1
                # if noise_count > 20: break  # limit data to speed up test

    landmark_features = np.array(all_features)
    feature_labels = (np.array(all_labels))

    # determine accuracy of noisy images
    false_neg = noise_accuracy(noise, img_labels)
    accuracy = 1-len(false_neg)/len(img_paths)
    end = time.time()

    print(data_count, "data points and", noise_count, "noisy images")

    # store noisy image labels
    f = open(os.path.join('out','noise'), "w+")
    f.write("%d noisy images were detected in %0.2f min:\r\n"
            "Accuracy = %0.2f with grayscale and gamma correction\r\n\n"
            % (noise_count, (end-start)/60, accuracy))
    [f.write("%s, " % img_num) for img_num in noise]
    f.write("\r\n\n\n%d false negatives were found (%0.2f FNR):\r\n\n"
            % (len(false_neg), len(false_neg)/noise_count))
    [f.write("%s, " % FN) for FN in false_neg]
    f.close()

    return landmark_features, feature_labels


# Test face detection accuracy

def noise_accuracy(indx, noise_labels):
    f_neg = []
    for i in range(len(indx)):
        checker = np.sum(noise_labels[indx[i]])
        # iff all 4 labels are -1, img is noise
        if checker > -4:
            f_neg.append(indx[i])

    return f_neg


# Localise face + predict shape using dlib and OpenCV
# via a pre-trained Histogram of Oriented Gradients (HOG)

def get_landmarks(img):
    # resize image + convert to grayscale
    img = img.astype('uint8')

    gamma = 3.0
    gamma_inv = 1.0 / gamma
    table = np.array([((i / 255.0) ** gamma_inv) * 255
                      for i in np.arange(0, 256)]).astype("uint8")

    # gamma correction using lookup table
    gam = cv2.LUT(img, table)

    gray = cv2.cvtColor(gam, cv2.COLOR_BGR2GRAY)
    gray = gray.astype('uint8')

    # localise faces in grayscale
    rects = detector(gray, 1)
    n_faces = len(rects)

    if n_faces == 0:
        return None, img

    face_areas = np.zeros((1, n_faces))
    face_shapes = np.zeros((136, n_faces), dtype=np.int64)

    # loop through dlib's detections
    for (i, rect) in enumerate(rects):
        # apply shape predictor: get landmarks
        temp_shape = predictor(gray, rect)
        temp_shape = shape2np(temp_shape)

        # convert dlib rect to bounding box
        (x, y, w, h) = rect2bb(rect)
        face_shapes[:, i] = np.reshape(temp_shape, [136])
        face_areas[0,i] = w * h

    # find + keep largest face
    face_out = np.reshape(np.transpose(face_shapes[:, np.argmax(face_areas)]), [68, 2])

    return face_out, img


# Util functions

def rect2bb(rect):
    # convert dlib detector's rect to bounding box for convenience
    x = rect.left()
    y = rect.top()
    w = rect.right() - x
    h = rect.bottom() - y

    return (x, y, w, h)


def shape2np(shape, dtype="int"):
    # init list of x,y coordinates
    xy = np.zeros((shape.num_parts, 2), dtype=dtype)

    # convert all landmarks to x,y 2-tuple
    for i in range(shape.num_parts):
        xy[i] = (shape.part(i).x, shape.part(i).y)

    return xy


# Store features and labels

def update_features():
    features, labels = label_features()

    # reshape features to 2D matrix
    m = features.shape
    features = np.reshape(features, (m[0], m[1] * 2)).astype(float)

    # convert to dataframe + store in csv
    df = pd.DataFrame(features)
    df.to_csv(os.path.join('out', 'features.csv'))

    dl = pd.DataFrame(labels)
    dl.to_csv(os.path.join('out', 'labels.csv'))

    return features, labels


# update_features()
