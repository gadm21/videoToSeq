
from utils import *
from VideoHandler import VideoHandler
from vocab import Vocab

#import matplotlib.pyplot as plt
import numpy as np
import os
import sys


from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.layers import Dropout, Flatten, RepeatVector, Activation, Concatenate
from tensorflow.keras.layers import Embedding, Conv2D, MaxPooling2D, LSTM, GRU, BatchNormalization
from tensorflow.keras.layers import TimeDistributed, Dense, Input, GlobalAveragePooling2D, Bidirectional
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.optimizers.schedules import PolynomialDecay
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.preprocessing import image
from tensorflow.keras.regularizers import l2


import tensorflow as tf
import tensorflow.keras.backend as K



class LearningRateDecay:
	def plot(self, epochs, title="Learning Rate Schedule"):
		# compute the set of learning rates for each corresponding
		# epoch
		lrs = [self(i) for i in epochs]
		# the learning rate schedule
        
		#plt.style.use("ggplot")
		#plt.figure()
		#plt.plot(epochs, lrs)
		#plt.title(title)
		#plt.xlabel("Epoch #")
		#plt.ylabel("Learning Rate")
        

class StepDecay(LearningRateDecay):
	def __init__(self, initAlpha=0.01, factor=0.25, dropEvery=10):
		# store the base initial learning rate, drop factor, and
		# epochs to drop every
		self.initAlpha = initAlpha
		self.factor = factor
		self.dropEvery = dropEvery
	def __call__(self, epoch):
		# compute the learning rate for the current epoch
		exp = np.floor((1 + epoch) / self.dropEvery)
		alpha = self.initAlpha * (self.factor ** exp)
		# return the learning rate
		return float(alpha)













class VModel:

    def __init__(self, params):

        self.callbacks = []
        self.params = params
        self.model_path = params['model_path']
        K.clear_session()

        self.build_model()
        


    def build_model(self):
        
        conv2d_1 = Conv2D(5, 7, dilation_rate=3, activation='relu')
        conv2d_2 = Conv2D(7, 5, dilation_rate=2, activation='tanh')
        conv2d_3 = Conv2D(10, 3, dilation_rate=2, activation='relu')
        conv2d_4 = Conv2D(15, 3,  activation='tanh')
        

        dense_i = Dense(1024, kernel_initializer='random_normal') 
        gru_i = GRU(1024, return_sequences=False, kernel_initializer='random_normal') 

        v_model_input = Input(shape = ( self.params['FRAME_SIZE'], self.params['FRAME_SIZE'], 1))

        v_model = MaxPooling2D(2, 2)(conv2d_1(v_model_input))
        v_model = Dropout(0.2)(v_model) 
        v_model = BatchNormalization()(v_model)

        v_model = MaxPooling2D(2, 2)(conv2d_2(v_model))
        v_model = Dropout(0.2)(v_model) 
        v_model = BatchNormalization()(v_model)

        v_model = MaxPooling2D(2, 2)(conv2d_3(v_model))
        v_model = Dropout(0.2)(v_model) 
        v_model = BatchNormalization()(v_model)

        v_model = conv2d_4(v_model) 
        v_model = Flatten()(v_model)

        model = Model(inputs=v_model_input, outputs=v_model)
        

        lr_polynomial_decay = PolynomialDecay()
        opt = RMSprop(lr=0.01, rho=0.9, epsilon=1e-8, decay=0.5)
        model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
        
        self.frame_model = model 
        
        
        

    def plot_model(self):
        tf.keras.utils.plot_model(self.model, 'visuals/model.png', show_shapes=True, show_layer_names=False) 
        tf.keras.utils.plot_model(self.model, 'visuals/more_specific_model.png', show_shapes=True) 




if __name__ == '__main__':
    params = read_yaml() 
    vmodel = VModel(params)
    vmodel.frame_model.summary()
