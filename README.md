# AMLSassignment
Applied Machine Learning Systems ELEC0132 Assignment

Maryam Habibollahi (SN: 15000241)

## Assignment tasks
* Detection and removal of noisy images
* Training, validation and testing subsets division
* Train ML models to perform
    - _Binary_
    1. Emotion recognition (smile/!smile)
    2. Age identification (young/old)
    3. Glasses detection (with/without) 
    4. Human detection (real/avatar)
    - _Multiclass_
    5. Hair colour recognition (ginger, blond, brown, grey, black, bald)


_[Dataset folder](https://drive.google.com/drive/folders/1NgP2jQakFHibIhpevDLshodWw-L52yXi)_

## How to compile and use code

### Required libraries

To run the classification models, the following libraries are required: Scikit-learn, Keras, dlib, OpenCV.

The libraries NumPy, Pandas, Time, OS, and MatplotLib are also required for data processing, file handling, and data representation.

### Python files

* landmarks.py implements funtions on Haar Cascode, HOG, and Deep Learning-based face detectors for obtaining image landmarks from detected faces
* classification.py includes the required functions for SVM and MLP implementation, as well as cross-validation testing
* utils.py provides the utility functions for handling files and data used in landmarks.py and classification.py
* testing.py runs the detection and classification functions and stores the data to file
* models/lenet.py implements the LeNet architecture for the multiclass classification task

### folders

- out/ includes all the results from the tests (see README.md in out/)
- models/ contains face detection models and the LeNet architecture setup file
