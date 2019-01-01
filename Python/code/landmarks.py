import os
import cv2
import dlib
import time
import numpy as np
import utils as ut
from keras.preprocessing import image

# Global vars
test = False
detectors = {"HOG": 1, "HaarCas": 2, "DNN": 3, "CNN": 4, "Test": 5}


# Run get_landmarks on all images
# Return features + labels in np array
# Store noisy images (get_landmarks = None)
def label_features(det):
    start = time.time()

    # load image + label data
    img_paths = [os.path.join(ut.img_dir, l) for l in os.listdir(ut.img_dir)]
    # order items
    img_paths = sorted(sorted(img_paths), key=len, reverse=False)

    labels_file = open(ut.labels_dir, 'r')
    lines = labels_file.readlines()
    # Store all labels in a dictionary
    img_labels = {line.split(',')[0] : [int(line.split(',')[col]) for col in range(1, 6)] for line in lines[2:]}

    if os.path.isdir(ut.img_dir):
        all_features = []
        all_labels = []
        noise = []
        data_count = 0

        for img_path in img_paths:
            img_name = img_path.split('.')[2].split('/')[-1]

            img = image.img_to_array(image.load_img(img_path,
                                                    target_size=None,
                                                    interpolation='bicubic'))
            img_features, _ = get_landmarks(det, img)
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
    ut.report_noise(det, end-start, noise, img_labels, img_paths)

    return landmark_features, feature_labels


# Select and run face detector
# Predict shape landmarks using dlib
# Return landmark features + img
def get_landmarks(det, img):
    # select detector
    case = detectors[det]
    if case == 1: rects, gray = HOG_detect(img)
    elif case == 2: rects, gray = HaarCas_detect(img)
    elif case == 3: rects, gray = DNN_detect(img)
    elif case == 4: rects, gray = CNN_detect(img)
    else: rects, gray = CNN_detect(img)

    n = len(rects)

    if n == 0:
        return None, img

    face_areas = np.zeros((1, n))
    face_shapes = np.zeros((136, n), dtype=np.int64)

    # loop through dlib's detections
    for (i, rect) in enumerate(rects):
        # apply shape predictor: get landmarks
        temp_shape = ut.predictor(gray, rect)
        temp_shape = ut.shape2np(temp_shape)

        # convert dlib rect to bounding box
        (x, y, w, h) = ut.rect2bb(rect)
        face_shapes[:, i] = np.reshape(temp_shape, [136])
        face_areas[0, i] = w * h

    # find + keep largest face
    face_out = np.reshape(np.transpose(face_shapes[:, np.argmax(face_areas)]), [68, 2])

    return face_out, img


# Localise face + predict shape using dlib and OpenCV
# via a pre-trained Histogram of Oriented Gradients (HOG)
# Pre-processes, detects + returns face dlib rects
def HOG_detect(img):
    # image resize, gamma + grayscale
    img = img.astype('uint8')
    gam = ut.correct_gamma(img, 3.0)
    gray = cv2.cvtColor(gam, cv2.COLOR_BGR2GRAY)
    gray = gray.astype('uint8')

    # localise faces in grayscale
    rects = ut.hog_detector(gray, 1)

    return rects, gray


# Detect face area using Haar-cascade
# Return all faces' dlib rects
def HaarCas_detect(img):
    # resize image + convert to grayscale
    img = img.astype('uint8')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = gray.astype('uint8')

    # localise faces in grayscale
    rects = ut.face_cascade.detectMultiScale(gray)
    dlib_rects = []
    # convert to dlib rect for predict
    for (x, y, w, h) in rects:
        dlib_rects.append(dlib.rectangle(int(x), int(y), int(x + w), int(y + h)))

    return dlib_rects, gray


# Detect face area based on Single-Shot-Multibox
# detector using ResNet-10 Architecture as backbone
def DNN_detect(img):
    img = img.astype('uint8')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    blob = cv2.dnn.blobFromImage(img, 1.0, (300, 300), [104, 117, 123], False, False)

    rects = []
    conf_thresh = 0.5
    frameWidth = 1  # CHECK
    frameHeight = 1  # CHECK

    ut.net.setInput(blob)
    detections = ut.net.forward()
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_thresh:
            x1 = int(detections[0, 0, i, 3] * frameWidth)
            y1 = int(detections[0, 0, i, 4] * frameHeight)
            x2 = int(detections[0, 0, i, 5] * frameWidth)
            y2 = int(detections[0, 0, i, 6] * frameHeight)

            conf_thresh = confidence
            rects.append(dlib.rectangle(int(x1), int(y1), int(x2), int(y2)))

    return rects, gray


# Detect face using a Maximum-Margin Object
# Detector ( MMOD ) with CNN-based features
def CNN_detect(img):
    img = img.astype('uint8')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    rects = []
    faceRects = ut.dnnFaceDetector(gray, 0)
    for faceRect in faceRects:
        x1 = faceRect.rect.left()
        y1 = faceRect.rect.top()
        x2 = faceRect.rect.right()
        y2 = faceRect.rect.bottom()
        rects.append(dlib.rectangle(int(x1), int(y1), int(x2), int(y2)))

    return rects, gray


