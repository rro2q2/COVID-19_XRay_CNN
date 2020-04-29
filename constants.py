import numpy as np
import matplotlib.pyplot as plt
from imutils import paths
from collections import Counter
from sklearn.model_selection import train_test_split
import tensorflow as tf
import cv2
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications import densenet
from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from tensorflow.keras.optimizers import *
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from tensorflow.keras.utils import to_categorical
import keras
from keras import regularizers
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, Callback
import timeit
import pandas as pd

## Global hyperparameters ##
learning_rate = 1e-3
epochs = 10
batch_size = 10
num_classes = 2

