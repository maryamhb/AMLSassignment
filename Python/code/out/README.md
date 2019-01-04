This folder contains _performance measures_ as well as features and labels of each method 
for convenient access during testing or further training stages. A summary of the statistics
on the performance of the applied techniques for face detection and feature classification 
is presented below.

## Dataset info

| Total images | Faces | Noise | Prevalence |
|:------------:|:-----:|:-----:|:----------:|
|    5,000     |  4565 |  435  |   0.913    |

## Face detection
Four different methods from the OpenCV and dlib libraries are tested on the dataset and compared against the labels, 
as listed below:

|                 Method               | Accuracy |  TPR  | TNR | FPR | FNR  | Time (min) |
| ------------------------------------ |:--------:|:-----:|:-----:|:-----:|:----:|:-----:|
| Haar Cascade detector                |   0.78   |  0.77 | 0.40 | 0.03 | 0.72 | 1.88  |
| Histogram of Oriented Gradients      |   0.97   |  0.97 | 1.00 | 0.00 | 0.33 | 2.79  |
| Single-Shot-Multibox detector (DNN)  |   0.99   |  0.98 | 1.00 | 0.00 | 0.14 | 4.55  |
| Maximum-Margin Object Detector (CNN) |   0.98   |  0.98 | 1.00 | 0.00 | 0.18 | 10.85 |


## Binary tasks

The average inference accuracies of the prediction on the test set (20% of the dataset) for each classification method are listed below:

### Task 1

Emotion detection - smiling or !smiling

|              Method            |   Accuracy  |  Recall (TPR) | Specificity (TNR) | Precision |
| ------------------------------ |:-----------:|:-------------:|:-----------------:|:---------:|
| Linear regression SVM          |     0.92    |     0.96      |        0.79       |   0.94    |
| Polynomial kernel SVM          |     0.92    |     0.96      |        0.79       |   0.94    |
| Sigmoid kernel SVM             |     0.78    |     0.78      |        0.00       |   1.00    |
| RBF kernel SVM                 |     0.91    |     0.95      |        0.79       |   0.94    |
| Relu-based MLP (3, 2)       |     0.93    |     0.95      |        0.86       |   0.96    |

### Task 2

Age identification - young or old

|              Method            |   Accuracy  |  Recall (TPR) | Specificity (TNR) | Precision |
| ------------------------------ |:-----------:|:-------------:|:-----------------:|:---------:|
| Linear regression SVM         |   0.78   |     0.78      |        0.00       |   1.00    |
| Polynomial kernel SVM         |   0.79   |     0.82      |        0.54       |   0.93    |
| Sigmoid kernel SVM            |   0.78   |     0.78      |        0.00       |   1.00    |
| RBF kernel SVM                 |   0.78   |     0.78      |        0.00       |   1.00    |
| Relu-based MLP (3, 2)       |     0.79    |     0.81      |        0.60       |   0.96    |

### Task 3

Glasses detection - with or without

|              Method            |   Accuracy  |  Recall (TPR) | Specificity (TNR) | Precision |
| ------------------------------ |:-----------:|:-------------:|:-----------------:|:---------:|
| Linear regression SVM         |   0.85   |     0.80      |        0.87       |    0.64   |
| Polynomial kernel SVM         |   0.85   |     0.76      |        0.87       |    0.67   |
| Sigmoid kernel SVM            |   0.72   |     0.00      |        0.72       |    0.00   |
| RBF kernel SVM                 |   0.85   |     0.90      |        0.84       |    0.52   |
| Relu-based MLP (3, 2)       |     0.86    |     0.81      |        0.88       |   0.66    |

### Task 4

Human classification - real or !real

|              Method            |   Accuracy  |  Recall (TPR) | Specificity (TNR) | Precision |
| ------------------------------ |:-----------:|:-------------:|:-----------------:|:---------:|
| Linear regression SVM         |   0.97   |     0.98      |       0.96        |    0.96   |
| Polynomial kernel SVM         |   0.98   |     0.98      |       0.97        |    0.96   |
| Sigmoid kernel SVM            |   0.54   |     0.00      |       0.54        |    0.00   |
| RBF kernel SVM                 |   0.97   |     0.96      |       0.97        |    0.97   |
| Relu-based MLP (3, 2)        |     0.97    |     0.99      |        0.96       |   0.95    |


Time taken for each classifier to run (on all 4 tasks):

|  Method  | Linear |  Poly | Sigmoid | RBF  |
| -------- |:------:|:------:|:------:|:----:|
| Time     |  2.87  | 44.23  |  0.11  | 0.08 |

## Multiclass task

### Task 5

Hair colour detection - blond, ginger, brown, grey, black, bald
