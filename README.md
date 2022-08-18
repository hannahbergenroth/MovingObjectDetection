# Moving Object Detection
Project developed in spring 2021 

The code is the implementation of my master's thesis Use of Thermal Imagery for Robust Moving Object Detection - Download paper [here](https://liu.diva-portal.org/smash/record.jsf?pid=diva2%3A1578091&dswid=-8919)

This work proposes a system that utilizes both infrared and visual imagery to create a more robust object detection and classification system. The system consists of two main parts: a moving object detector and a target classifier. The first stage detects moving objects in visible and infrared spectrum using background subtraction based on Gaussian Mixture Models. Low-level fusion is performed to combine the foreground regions in the respective domain. For the second stage, a Convolutional Neural Network (CNN), pre-trained on the ImageNet dataset is used to classify the detected targets into one of the pre-defined classes; human and vehicle. 

## Technologies
Project built with Python 3.6

For virtualenv to install all files in the requirements.txt file.
1. cd to the project directory where requirements.txt is located
2. activate your virtualenv
3. run `pip install -r requirements.txt` 

## Downloads

Download the VGG16 pretrained weights manually from [here](https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5) or:
```
$ cd model/weights
$ wget https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5 -O vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5
```
## Directory Structure

```
project
│
├───data
│   └── ...
│
├───detector
│   ├── Training.py
│   ├── Training_RGB.py
│   └── Training_T.py
│   
├───evaluate
│   ├── augmentation.py
│   ├── iou.py
│   ├── maskExtraction.py
│   └── measurements.py
│   
├───ground_truth
│   └── ...
│ 
├───model
│   ├───my_model
│   └───weights
│
├───training_data
│   ├───people
│   │   └── ...
│   └───vehicle
│       └── ...
│
├── README.md
├── network.py
├── predictions.py
└── requirements.txt
```
