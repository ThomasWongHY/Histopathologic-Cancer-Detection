# Histopathologic Cancer Detection Project 

## Overview

This project aims to develop a machine learning model for the detection of cancer tissue in histopathologic scans of lymph node sections. The repository contains a series of Python scripts that preprocess the dataset and train a model using deep learning techniques.

## Dataset

Download the dataset using the following link: https://www.kaggle.com/c/histopathologic-cancer-detection/data 

Click on the “Download All” button to download the 4 files, once download is completed, unzip it and you can use the file to run our ipynb file provided. 

The dataset comprises high-resolution histopathologic scans where each image is labeled as 1 (presence of tumor tissue) or 0 (absence of tumor tissue).

- The `train_labels.csv` file contains the ground truth labels for the training images, where each image is marked as '1' for the presence of tumor tissue and '0' for its absence.

- The `sample_submission.csv` file is a template provided by Kaggle for submitting your predictions; it typically contains image IDs and a column for predicted labels.

- The `train` file contains all the images for training purposes and the `test` file contains all the images for testing purposes.

![image](https://github.com/ThomasWongHY/Histopathologic-Cancer-Detection/assets/86035047/447534ba-170f-4b3b-84a9-14c91aa0411c)

## Code Description

### Data Preprocessing

The dataset is preprocessed to prepare it for the machine learning model. This includes:

•	Loading image labels and sample submission data.

•	Checking for and handling missing values or duplicates.

•	Adding correct file extensions to image IDs.

•	Stratifying the dataset to maintain class balance when splitting into training and validation sets.

•	Augmenting image data to increase model robustness against overfitting.

### Image Loading and Augmentation

Custom functions are defined to load images in grayscale, resize them, and augment the training data. The following operations are performed:

•	Grayscale conversion and resizing to 96x96 pixels.

•	Augmentation techniques such as rotation, width and height shifts, shear, zoom, and horizontal flips.

•	Normalization of pixel values.

### Visualization

Multiple visualization functions are included to:

•	Display a random subset of images with their corresponding labels.

•	Show a subset of tumor and non-tumor images in grayscale.

•	Plot the distribution of tumor and non-tumor labels in the training set.

•	Visualize augmented grayscale images to demonstrate the effect of data augmentation.

•	Plot pixel value distributions (histograms) for a set of images.

### State-of-the-art Model: InceptionV3

Inception V3 is a CNN architecture designed for improved accuracy and efficiency by using parallel convolutions at different scales. It employs Factorization methods and dimension reduction to reduce complexity. Auxiliary Classifiers combat the vanishing gradient problem, while Batch Normalization accelerates training. It utilizes RMSprop Optimizer with a decay term and Label Smoothing regularization for enhanced convergence and generalization.

By applying transfer learning, it can be fit into our domain (Cancer Detection). So I added 2 dense layers with a dropout with 0.2 on top of the Inception V3 model.

### Result
Surprisingly, the new model after transfer learning from Inception V3 model achieved 100% accuracy in both validation dataset and testing dataset. Both results were predicted by normalized images.

![image](https://github.com/ThomasWongHY/Histopathologic-Cancer-Detection/assets/86035047/4ce0ff6d-f8ff-4b7d-aafb-e435afba9fa3)

![image](https://github.com/ThomasWongHY/Histopathologic-Cancer-Detection/assets/86035047/ec48e050-bb78-4437-9cf9-4e3ea58cb8a4)
