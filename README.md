Alzheimer's Disease Classifier using AI

Objective

The objective of this work is to develop an efficient and accurate Alzheimer's disease classifier using artificial intelligence that enables the detection and automatic classification of 3D magnetic resonance images from the American ADNI (Alzheimer’s Disease Neuroimaging Initiative) database, specifically using ADNI1_Complete_3Yr_1.5T_7_05_2024.
Background

Before starting this project, extensive research was conducted on the state of the art by reviewing numerous academic papers. These papers primarily focused on 2D classifiers that used 3D images but only from a specific slice, usually centered on the hippocampus region. In my case, to determine where a neural network focuses its attention for medical study in the early detection of Alzheimer's or Mild Cognitive Impairment (MCI), I processed the entire 3D image.
Data Preparation
Dataset Overview

As mentioned previously, the dataset contained 2142 images of patients taken at different control points. Upon reviewing the data, one of the issues encountered was the method of acquisition of the NiFTI images. A significant portion of the data was taken with a Siemens scanner, but a substantial percentage was acquired using scanners from other companies such as GE Medical Systems and Philips. This variation in scanner models resulted in a heterogeneous dataset with non-equivalent NiFTI image resolutions.
Dataset Cleaning

To address this issue, images whose 'x' and 'y' coordinates were not equal to 256 were removed, as well as some residual image sets. This left a dataset with a total of 1302 NiFTI images. For the 'z' dimension, which manages depth, it was decided to retain the slices closest to the center of the image for all images.
Preprocessing Techniques
Homogenization

The preprocessing section discusses how the issue of diverse resolutions was further managed.
Diagnostic Group Classification

Once the dataset was cleaned, classification was performed considering the three diagnostic groups (AD, CN, MCI) with the new total of 1302 NiFTI images.
Dataset Balancing

As observed, the classes within the dataset were excessively imbalanced, which could cause future learning issues for the model. Therefore, balancing was performed by equalizing the classes with more data to match the AD class, which was also reduced to avoid too many control points from the same patient.

This adjustment ensured correct data generalization due to equity in prediction, providing sufficient examples of each class during training. It also prevented bias towards the majority class, ensuring that minority classes were not ignored. With the balanced dataset, the model evaluation would be more accurate, reflecting the model's performance across all classes, which is of crucial importance in this case.
Applied Preprocessing Techniques

In the context of medical imaging, registration and normalization are crucial processes for standardizing images acquired through different means. By adjusting intensity values, data consistency improves, leading to better model learning. The Min-Max technique was used, rescaling intensity values to be between 0 and 1.

Segmentation techniques were applied to extract the most relevant tissues from the original NiFTI image for Alzheimer's disease detection. Attempts were made to extract gray matter, white matter, and cerebrospinal fluid. Otsu's thresholding was applied to calculate the optimal threshold value using the method developed by Nobuyuki Otsu in 1979, which automatically selects the optimal value to separate the desired pixels.
Methodology

    Neural Network Architecture:
        The neural network architecture was designed to handle 3D images effectively. (Provide details of the architecture if possible.)

    Training:
        The model was trained using the processed and augmented dataset.
        Regularization techniques were applied to prevent overfitting.

    Validation:
        The validation set was used to tune hyperparameters and evaluate the model's performance during training.

    Testing:
        The final evaluation was conducted on the test set to measure the model's accuracy, precision, recall, and other relevant metrics.

Results

    The classifier achieved an accuracy of X% on the test set.
    Detailed results can be found in the results section. (Include tables, graphs, or any relevant metrics.)
