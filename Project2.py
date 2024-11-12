# -*- coding: utf-8 -*-
"""
Created on Sun Nov 10 18:05:14 2024

@author: arjun


Arjun Tripathi
501 021 964
AER850 - Project 2
"""

"Step 1: Data Processing"

"Import Librarires"

import pandas as pd
import numpy as np
import tensorflow as tf
from tenorflow import keras
from keras import layers
from keras import models
from keras import imagedatasetfromdirectory
from keras import ImageDataGenerator

"Define Input Image Shape"
Img_width = 500
Img_height = 500
Img_channel = 3

"Define Train and Validation Data Directory"
Train_direct = "Project 2\AER850Project2\Project 2 Data\Data\train"
Validation_direct = "Project 2\AER850Project2\Project 2 Data\Data\valid"

"Data Augmentation For Train & Validation"

Train_datagenerator = ImageDataGenerator (
    rescale = 1./255,
    shear_range = 0.2,
    zoom_range = 0.2,
    horizontal_flip = True,
    )

Validation_datagenerator = ImageDataGenerator (rescale = 1./255)

"Train & Validation Image Generators"

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



