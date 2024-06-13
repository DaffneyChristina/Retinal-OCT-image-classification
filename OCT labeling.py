#!/usr/bin/env python
# coding: utf-8

# In[2]:


import warnings
warnings.filterwarnings('ignore')
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
from sklearn.metrics import classification_report , confusion_matrix , accuracy_score , auc
from sklearn.model_selection import train_test_split


# In[4]:


pip install opencv-python-headless


# In[7]:


pip install tensorflow


# In[9]:


pip install tensorflow_hub


# In[10]:


import cv2
from PIL import Image 
import tensorflow as tf
from tensorflow import keras
from keras import Sequential
from keras.layers import Input, Dense,Conv2D , MaxPooling2D, Flatten,BatchNormalization,Dropout
from tensorflow.keras.preprocessing import image_dataset_from_directory
import tensorflow_hub as hub


# In[11]:


import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2 , ResNet152
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Sequential


# In[13]:


path = "C:/Users/daffn/Downloads/archive/OCT2017"


# In[16]:


path = "C:/Users/daffn/Downloads/archive/OCT2017/train/CNV/CNV-417468-8.jpeg"
image = plt.imread(path)
image.shape


# In[17]:


train = image_dataset_from_directory("C:/Users/daffn/Downloads/archive/OCT2017/train",color_mode="grayscale",
                                    image_size=(224,224),batch_size=512,shuffle=True)


# In[18]:


test = image_dataset_from_directory("C:/Users/daffn/Downloads/archive/OCT2017/test", color_mode="grayscale",
                                    image_size=(224,224),batch_size=512,shuffle=True)


# In[19]:


val = image_dataset_from_directory("C:/Users/daffn/Downloads/archive/OCT2017/val", color_mode="grayscale",
                                    image_size=(224,224),batch_size=512,shuffle=True)


# In[20]:


class_labels = train.class_names
class_labels


# In[21]:


for img_batch, label in train.take(1):
    print(img_batch.shape)
    print(label)
    break


# In[22]:


plt.figure(figsize=(30,55))
for img_batch, label in train.take(1):
    for i in range(64):
        plt.subplot(11,6,i+1)
        plt.imshow(img_batch[i])
        plt.title(f"Actual Label:{class_labels[label[i]]}")
        plt.axis("off")
    break


# In[24]:


# Create an ImageDataGenerator with scaling and data augmentation
datagen = ImageDataGenerator(
    rescale=1.0/255.0,  # Scale pixel values to the range [0, 1]
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True
)

# Load and preprocess the images using the generator
batch_size = 512  # Adjust the batch size according to your needs
image_size = (224,224)

train_data_generator = datagen.flow_from_directory(
    "C:/Users/daffn/Downloads/archive/OCT2017/train",
    target_size=image_size,
    batch_size=batch_size,
    class_mode='sparse',
    #subset='training',  # Specify 'training' to get the training set
    shuffle=True
)


# In[26]:


test_data_generator = datagen.flow_from_directory(
    "C:/Users/daffn/Downloads/archive/OCT2017/test",
    target_size=image_size,
    batch_size=batch_size,
    class_mode='sparse',
    #subset='validation',  # Specify 'validation' to get the validation set
    shuffle=True
)


# In[27]:


val_data_generator = datagen.flow_from_directory(
    "C:/Users/daffn/Downloads/archive/OCT2017/val",
    target_size=image_size,
    batch_size=batch_size,
    class_mode='sparse',
    #subset='validation',  # Specify 'validation' to get the validation set
    shuffle=True
)


# In[30]:


train_data_generator.class_indices


# In[ ]:




