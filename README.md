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

* Classification models: Scikit-learn, Keras, dlib, and OpenCV.

* Data processing, storage, and representation: NumPy, Pandas, Time, OS, and MatplotLib.

### Python files

* landmarks.py 

Implements funtions on Haar Cascode, HOG, and Deep Learning-based face detectors for obtaining image landmarks from detected faces.

* classification.py 

Includes the required functions for SVM and MLP implementation, as well as cross-validation testing.

* utils.py 

Provides the utility functions for handling files and data used in landmarks.py and classification.py.

* testing.py 

Calls the detection and classification functions and stores the results to csv. This file includes one function for binary tasks 1-4 and two for multiclass task 5. The landmarks previously stored in out/ are used by default to save time. To re-run the face detector, uncomment function update_features()

* models/lenet.py 

Implements the LeNet architecture for the multiclass classification task

### Folders in Python

- code/out/ 

Includes all the results from the tests (see out/README.md)
Face detector features are stored in Face_detection/ for convenience 

- code/models/

Contains face detection models and the LeNet architecture setup file
