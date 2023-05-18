# Term_project-Scoliosis-Image-Classification
Scoliosis Image Classification by using Convolutional Neural Network

This project focuses on image classification using Convolutional Neural Networks (CNNs) for scoliosis detection. It provides a Python-based solution for training and evaluating CNN models to classify scoliosis images into different categories or classes.

# Table of Contents

# Project Description
1. Installation
2. Usage
3. Dataset
4. Model Training
5. Results
6. Project Description

Image Classification by using Convolutional Neural Network is a project aimed at building and training deep learning models for accurate image classification tasks. It utilizes the power of CNNs to automatically extract relevant features from images and make predictions based on those features. This approach is particularly effective for tasks such as object recognition, identifying patterns, or detecting specific features within images.
This project provides a starting point for developers and researchers interested in image classification and CNNs. It offers a codebase that can be extended or modified to fit specific image classification tasks, datasets, and requirements.

# Installation

To use this project, follow these steps:

1. import tensorflow as tf
2. import keras as tf
3. from tensorflow.keras.layers import Conv2D, MaxPooling2D
4. from tensorflow.keras.models import Sequential
5. from tensorflow.keras.layers import Dense, Flatten
6. import numpy as np
7. !pip install tf-nightly

# Usage

To train and evaluate the image classification model:

1. Prepare your dataset by organizing images into separate folders for each class/category.
2. Modify the configuration settings in config.py to suit your requirements, such as dataset paths, model hyperparameters, and training parameters.
3. Run the training script

# Dataset

2 classes; double and single curve
39 models

# Model Training

The image classification model is built using the TensorFlow and Keras libraries. The CNN architecture, hyperparameters, and training configurations can be modified in the codebase. By default, a standard CNN architecture is implemented, but feel free to experiment with different architectures and configurations to achieve better performance.

# Results

The training set was divided into 2 batches and the model took 1 second per batch to train.
The loss on the training set decreased over time, reaching a final value of 0.0199 at the end of the 30th epoch.
The accuracy on the training set increased over time, reaching a final value of 1.0 (100%) at the end of the 30th epoch.
The validation set was also evaluated after each epoch, with the loss and accuracy metrics being recorded.
The loss on the validation set increased over time, reaching a final value of 2.4527 at the end of the 30th epoch.
The accuracy on the validation set fluctuated during training, but had a final value of 0.7143 (71.43%) at the end of the 30th epoch.
Overall, this suggests that the model performed well on the training set, achieving high accuracy and low loss. However, the model did not generalize as well to the validation set, with higher loss and lower accuracy on this set. This could indicate overfitting, where the model has memorized the training set and is not able to generalize well to new data. Further tuning and evaluation may be necessary to improve the performance of the model.





