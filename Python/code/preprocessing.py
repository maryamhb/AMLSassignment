import os
import numpy as np

# Data directory
img_path = os.path.join('..','dataset')
labels_path = os.path.join('..','attribute_list.csv')

# Load data
images = [os.path.join(img_path, l) for l in os.listdir(img_path)]
labels_file = open(labels_path, 'r')
lines = labels_file.readlines()


'''

Image manipulation libraries:
 - PIL (Pillow)
 - OpenCV (cv2)
 - scikit-image (skimage)
 
Pre-processing techniques:
 - Subtract mean intensity & divide by sd
 - Gamma correction (power-law equalisation)
 - Colour space transformation (RGB-LAB)
 
Feature extraction methods:
 - Haar-like features (Viola & Jones)
 - Histogram of Oriented Gradients (HOG)
 - Scale-Invariant Feature Transform (SIFT)
 - Speeded Up Robust Feature (SURF)
 
'''