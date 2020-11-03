
from utils import * 

import numpy as np 
import os 
import sys 
from vocab import Vocab

from keras.applications import ResNet50 
from keras.applications.resnet50 import preprocess_input 

from keras.layers import Dropout, Flatten, RepeatVector, Activation 
from tensorflow.keras.layers import Embedding, Conv2D, MaxPooling2D, LSTM, GRU, BatchNormalization, Merge
from keras.layers import TimeDistributed, Dense, Input, GlobalAveragePooling2D, Bidirectional
from keras.models import Model, Sequential 
from keras.optimizers import RMSprop
from keras.preprocessing import image 
from keras.regularizers import l2

import keras.backend as K 
import tensorflow as tf 




print("version:", tf.keras.__version__)





class VModel :

    def __init__(self, params):

        pass