
from utils import *
from VideoHandler import VideoHandler
from vocab import Vocab

import matplotlib.pyplot as plt
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
		plt.style.use("ggplot")
		plt.figure()
		plt.plot(epochs, lrs)
		plt.title(title)
		plt.xlabel("Epoch #")
		plt.ylabel("Learning Rate")

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

        if not self.params['cutoff_only']:
            self.build_mcnn()
        self.build_cutoff_model()
        

    def train_model(self):
        K.set_learning_phase(1)

    def build_mcnn(self):
        if self.params['learning']:
            self.train_model()
        log('debug', "creating model (CNN cutoff) with Vocab size: %d" % self.params['VOCAB_SIZE'])

        # _____________________________________________________________________________________
        dense_1 = Dense(200, kernel_initializer='random_normal', activation='relu')

        c_model_input = Input(shape= (self.params['CAPTION_LEN'],))
        c_model_embeds = Embedding(self.params['VOCAB_SIZE'], self.params['OUTDIM_EMB'])(c_model_input)
        c_model_2nd = TimeDistributed(dense_1)(c_model_embeds) 
        c_model_final = LSTM(128, return_sequences=True, kernel_initializer='random_normal')(c_model_2nd) 
        #c_model = Model(inputs=c_model_input, outputs= c_model_final, name='caption_model') 
        #print(c_model.summary()) 
        #tf.keras.utils.plot_model(c_model, 'c_model.png', show_shapes=True) 
        # _____________________________________________________________________________________
        

        '''
        # ___________________________________________________________________________________
        a_model_input = Input(shape=( self.params['AUDIO_TimeSample'], self.params['AUDIO_n_mfcc']))
        a_model_2nd = GRU(128, dropout=0.2, recurrent_dropout=0.2, return_sequences=True)(a_model_input) 
        a_model_2nd_normed = BatchNormalization()(a_model_2nd) 
        a_model_3rd = GRU(64, dropout=0.2, recurrent_dropout=0.2, return_sequences=True)(a_model_2nd_normed) 
        a_model_3rd_normed = BatchNormalization()(a_model_3rd) 
        a_model_flattened = Flatten()(a_model_3rd_normed) 
        a_model_final = RepeatVector(self.params['CAPTION_LEN']+1)(a_model_flattened) 
        #a_model = Model(inputs=a_model_input, outputs=a_model_final, name='audio_model') 
        #print(a_model.summary()) 
        #tf.keras.utils.plot_model(a_model, to_file='a_model.png', show_shapes=True) 
        # ___________________________________________________________________________________
        '''



        # _____________________________________________________________________________
        dense_i = Dense(1024, kernel_initializer='random_normal') 
        gru_i = GRU(1024, return_sequences=False, kernel_initializer='random_normal') 

        i_model_input = Input(shape = (self.params['FRAMES_LIMIT'], self.params['VIDEO_VEC']), name='video')
        i_model_2nd = TimeDistributed(dense_i)(i_model_input) 
        i_model_3rd = TimeDistributed(Dropout(0.2))(i_model_2nd) 
        i_model_4th = TimeDistributed(BatchNormalization(-1))(i_model_3rd)
        i_model_4th = Activation('tanh')(i_model_4th)
        i_model_5th = Bidirectional(gru_i)(i_model_4th) 
        i_model_final = RepeatVector(self.params['CAPTION_LEN'])(i_model_5th) 
        #i_model = Model(inputs=i_model_input, outputs=i_model_final, name='frames_model')
        #print(i_model.summary()) 
        #tf.keras.utils.plot_model(i_model, to_file='i_model.png', show_shapes=True) 
        # _____________________________________________________________________________
        
        lstm_concatted = LSTM(int(self.params['VOCAB_SIZE']*(3//4)), kernel_initializer='random_normal', recurrent_regularizer=l2(0.01))
        dense_concatted = Dense(self.params['VOCAB_SIZE'], kernel_initializer='random_normal')

        concatted = Concatenate(-1)([c_model_final, i_model_final])
        concatted = TimeDistributed(Dropout(0.2))(concatted)
        concatted = TimeDistributed(Dense(int(self.params['VOCAB_SIZE']//2), activation='relu'))(concatted) 
        concatted = lstm_concatted(concatted)
        concatted = dense_concatted(concatted) 
        concatted = Activation('softmax')(concatted) 


        schedule = StepDecay(initAlpha=1e-1, factor=0.25, dropEvery=10)
        model_checkpoint_callback = ModelCheckpoint(
            filepath = self.model_path,
            save_weights_only= True,
            monitor = 'accuracy',
            mode = 'max',
            save_best_only=True)
        self.callbacks.append(LearningRateScheduler(schedule))
        self.callbacks.append(model_checkpoint_callback)
        opt = RMSprop(lr=0.001, rho=0.9, epsilon=1e-8, decay=0)
        
        model = Model(inputs=[c_model_input, i_model_input], outputs=concatted)
        model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
        #tf.keras.utils.plot_model(model, to_file='total.png', show_shapes=True) 
        
        self.model = model 
        log('info', ' mcnn model created!') 
        
        
        

    def get_cutoff_shape(self):
        # ResNet
        shape = (None, 2048)
        log('debug', 'model cutoff outShape: %s' % str(shape))
        return shape

    def build_cutoff_model(self):
        base = ResNet50(include_top=False, weights='imagenet', pooling='avg') 
        self.co_model = Model(base.input, base.output) 

    def preprocess_partial_model(self, frames):
        return self.co_model.predict(preprocess_input(frames))
        

    def get_model(self):
        return self.model 
    
    def plot_model(self):
        tf.keras.utils.plot_model(self.model, 'visuals/model.png', show_shapes=True, show_layer_names=False) 
        tf.keras.utils.plot_model(self.model, 'visuals/more_specific_model.png', show_shapes=True) 




if __name__ == '__main__':
    params = read_yaml() 
    vmodel = VModel(params) 
    vmodel.co_model.summary()
    vmodel.plot_model() 
