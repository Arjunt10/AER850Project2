# -*- coding: utf-8 -*-
"""
Created on Sun Nov 10 18:05:14 2024

@author: arjun


Arjun Tripathi
501 021 964
AER850 - Project 2
"""

"STEP 1: Data Processing"

"Import Librarires"

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator

"Define Input Image Shape"
Img_width = 500
Img_height = 500
Img_channel = 3
Img_shape = (Img_width, Img_height, Img_channel)

"Define Train and Validation Data Directory"
Train_direct = r"Project 2 Data\Data\train"
Validation_direct = r"Project 2 Data\Data\valid"

"Data Augmentation For Train & Validation"

Train_datagenerator = ImageDataGenerator (
    rescale = 1./255,
    shear_range = 0.2,
    zoom_range = 0.2,
    horizontal_flip = True,
    )

Validation_datagenerator = ImageDataGenerator (rescale = 1./255)

"Applying Augmentation to Tran & Validation Datasets"

Train_generator = Train_datagenerator.flow_from_directory (
    
    Train_direct,
    target_size = (Img_width,Img_height),
    batch_size = 32,
    class_mode = 'categorical'   
    )

Validation_generator = Validation_datagenerator.flow_from_directory (
    
    Validation_direct,
    target_size = (Img_width,Img_height),
    batch_size = 32,
    class_mode = 'categorical'   
    )

"STEP 2: Neural Network Architecture Design"

"Creating The Convolutional Base"

model = models.Sequential ()

"Layer 1"
model.add(layers.Conv2D(32, (3,3), activation = 'relu', input_shape = Img_shape))
model.add(layers.MaxPooling2D((2,2)))

"Layer 2"

"Need to import LeakyRelu"
from keras.layers import LeakyRelu

model.add(layers.Conv2D(64, (3,3)))
model.add(LeakyRelu(negative_slope = 0.01))
model.add(layers.MaxPooling2D((2,2)))

"Layer 3"
model.add(layers.Conv2D(128, (3,3), activation = 'relu'))
model.add(layers.MaxPooling2D((2,2)))

"Flatten the Output from 3D to 1D"
model.add(layers.Flatten())

"Dense Layers"
model.add(layers.Dense(64, activation = 'relu'))
model.add(layers.dropout(0.5))
model.add(layers.Dense(3, activation = 'softmax'))

"STEP 3: Model Hyperparameter Analysis"

"Compiled Model"
model.compile(optimizer = 'adam', loss = "categorical_crossentropy", metrics = ["accuracy"])

"Model Summary"
model.summary()





