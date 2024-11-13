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

# import pandas as pd
# import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

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
model.add(layers.Input(shape = Img_shape))
model.add(layers.Conv2D(32, (3,3), activation = 'relu'))
model.add(layers.MaxPooling2D((2,2)))

"Layer 2"

"Need to import LeakyRelu"
from tensorflow.keras.layers import LeakyReLU

model.add(layers.Conv2D(64, (3,3)))
model.add(LeakyReLU(negative_slope = 0.01))
model.add(layers.MaxPooling2D((2,2)))

"Layer 3"
model.add(layers.Conv2D(128, (3,3), activation = 'relu'))
model.add(layers.MaxPooling2D((2,2)))

"Flatten the Output from 3D to 1D"
model.add(layers.Flatten())

"Dense Layers"
model.add(layers.Dense(64, activation = 'relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(3, activation = 'softmax'))

"STEP 3: Model Hyperparameter Analysis"

"Compiled Model"
model.compile(optimizer = 'adam', loss = "categorical_crossentropy", metrics = ["accuracy"])

"Model Summary"
model.summary()

"Model Training"
model_trained = model.fit(
    Train_generator,
    epochs = 15,
    validation_data = Validation_generator
    )

"STEP 4: Model Evaluation"
valid_loss, valid_accuracy = model.evaluate(Validation_generator)

accuracy = model_trained.history['accuracy']
validation_accuracy = model_trained.history['valid_accuracy']
loss = model_trained.history ['loss']
validation_loss = model_trained.history ['valid_loss']


"Plotting the Training vs Validation Accuracy Data"
plt.figure(figsize = (12,4))
plt.plot(accuracy, label = "Training Accuracy")
plt.plot(validation_accuracy, label = "Validation Accuracy")
plt.title ('Model Training and Validation Accuracy')
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()

"Plotting the Training vs Validation Loss Data"
plt.figure(figsize = (12,4))
plt.plot(loss, label = "Training Loss")
plt.plot(validation_loss, label = "Validation Loss")
plt.title ('Model Training and Validation Loss')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

plt.show()

model.save ("Initial Model - Project 2")
