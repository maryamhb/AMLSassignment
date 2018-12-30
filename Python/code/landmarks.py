import os
import cv2
import dlib
import time
import numpy as np
import utils as ut
from keras.preprocessing import image

# Test on data subset
test = True

# Data directory
img_dir = os.path.join('..', 'dataset')
labels_dir = os.path.join('..', 'attribute_list.csv')

# Face detector + landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')


# Run get_landmarks on all images
# Return features + labels in np array
# Store noisy images (get_landmarks = None)
def label_features(detector):
    start = time.time()

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

        for img_path in img_paths:
            img_name = img_path.split('.')[2].split('/')[-1]

            img = image.img_to_array(image.load_img(img_path,
                                                    target_size=None,
                                                    interpolation='bicubic'))
            img_features, _ = get_landmarks(detector, img)
            if img_features is not None:
                all_features.append(img_features)
                all_labels.append(img_labels[img_name])
                data_count += 1

            else:
                noise.append(img_name)
                if test and len(noise) > 10: break  # limit data to speed up test

    landmark_features = np.array(all_features)
    feature_labels = (np.array(all_labels))

    end = time.time()

    print(data_count, "data points and", len(noise), "noisy images")

    # report + store noise perf
    ut.report_noise(detector, end-start, noise, img_labels, img_paths)

    return landmark_features, feature_labels


# Select and run landmark
# detector based on title arg
# Return features + processed img
def get_landmarks(detector, img):
    landmark, img_out = HoG_landmarks(img)
    # landmark, img_out = Harr_cascade(img)

    return landmark, img_out


# Localise face + predict shape using dlib and OpenCV
# via a pre-trained Histogram of Oriented Gradients (HOG)
# Pre-processes, detects + returns largest face landmarks
def HoG_landmarks(img):
    # image resize, gamma + grayscale
    img = img.astype('uint8')

    gam = ut.correct_gamma(img, 3.0)

    gray = cv2.cvtColor(gam, cv2.COLOR_BGR2GRAY)
    gray = gray.astype('uint8')

    # localise faces in grayscale
    rects = detector(gray, 1)
    n = len(rects)

    if n == 0:
        return None, img

    face_areas = np.zeros((1, n))
    face_shapes = np.zeros((136, n), dtype=np.int64)

    # loop through dlib's detections
    for (i, rect) in enumerate(rects):
        # apply shape predictor: get landmarks
        temp_shape = predictor(gray, rect)
        temp_shape = ut.shape2np(temp_shape)

        # convert dlib rect to bounding box
        (x, y, w, h) = ut.rect2bb(rect)
        face_shapes[:, i] = np.reshape(temp_shape, [136])
        face_areas[0,i] = w * h

    # find + keep largest face
    face_out = np.reshape(np.transpose(face_shapes[:, np.argmax(face_areas)]), [68, 2])

    return face_out, img


# Collect face area using Harr-cascade
# detection in OpenCV as rect, return
# bounding box of largest face area
def Harr_cascade(img):
    # resize image + convert to grayscale
    img = img.astype('uint8')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # localise faces in grayscale
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    rects = face_cascade.detectMultiScale(gray)
    n = len(rects)

    if n == 0:
        return None, img

    face_areas = np.zeros((1, n))
    face_shapes = np.zeros((136, n), dtype=np.int64)

    # loop through dlib's detections
    for (i, rect) in enumerate(rects):
        # apply shape predictor: get landmarks
        temp_shape = predictor(gray, rect)
        temp_shape = ut.shape2np(temp_shape)

        # convert dlib rect to bounding box
        (x, y, w, h) = ut.rect2bb(rect)
        face_shapes[:, i] = np.reshape(temp_shape, [136])
        face_areas[0,i] = w * h

    # find + keep largest face
    face_out = np.reshape(np.transpose(face_shapes[:, np.argmax(face_areas)]), [68, 2])

    return face_out, img
