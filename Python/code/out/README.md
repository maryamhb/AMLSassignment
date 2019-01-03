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



## Multiclass tasks
