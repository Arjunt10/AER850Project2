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

"Applying Augmentation to Tran & Validation datasets"

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





