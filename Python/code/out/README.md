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

|                 Method               | Accuracy |  TPR  | FNR  | Time  |
| ------------------------------------ |:--------:|:-----:|:----:|:-----:|
| Haar Cascade detector                |   0.79   |  0.77 | 0.72 | 5.14  |
| Histogram of Oriented Gradients      |   0.97   |  0.96 | 0.25 | 2.98  |
| Single-Shot-Multibox detector (DNN)  |   0.99   |  0.98 | 0.14 | 4.55  |
| Maximum-Margin Object Detector (CNN) |   0.98   |  0.98 | 0.18 | 10.85 |


## Binary tasks



## Multiclass tasks
