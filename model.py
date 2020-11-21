
from utils import *
from VideoDataHandler import VideoDataHandler

import numpy as np
import os
import sys
from vocab import Vocab


from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.layers import Dropout, Flatten, RepeatVector, Activation, Concatenate
from tensorflow.keras.layers import Embedding, Conv2D, MaxPooling2D, LSTM, GRU, BatchNormalization
from tensorflow.keras.layers import TimeDistributed, Dense, Input, GlobalAveragePooling2D, Bidirectional
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing import image
from tensorflow.keras.regularizers import l2


import tensorflow as tf
import tensorflow.keras.backend as K


class VModel:

    def __init__(self, params):

        self.params = params

        if not self.params['cutoff_only']:
            self.build_mcnn()
        self.build_cutoff_model()
        

    def train_model(self):
        K.set_learning_phase(1)

    def build_mcnn(self):
        if self.params['learning']:
            self.train_model()
        log('debug', "creating model (CNN cutoff) with Vocab size: %d" %
            self.params['VOCAB_SIZE'])

        # _____________________________________________________________________________________
        c_model_input = Input(shape=(self.params['CAPTION_LEN'] + 1, self.params['OUTDIM_EMB']))
        dense_1 = Dense(512, kernel_initializer='random_normal')
        c_model_2nd = TimeDistributed(dense_1)(c_model_input) 
        c_model_final = LSTM(512, return_sequences=True, kernel_initializer='random_normal')(c_model_2nd) 
        #c_model = Model(inputs=c_model_input, outputs= c_model_final, name='caption_model') 
        #print(c_model.summary()) 
        #tf.keras.utils.plot_model(c_model, 'c_model.png', show_shapes=True) 
        # _____________________________________________________________________________________
        


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




        # _____________________________________________________________________________
        dense_i = Dense(1024, kernel_initializer='random_normal') 
        gru_i = GRU(1024, return_sequences=False, kernel_initializer='random_normal') 

        i_model_input = Input(shape = self.get_cutoff_shape())
        i_model_2nd = TimeDistributed(dense_i)(i_model_input) 
        i_model_3rd = TimeDistributed(Dropout(0.2))(i_model_2nd) 
        i_model_4th = TimeDistributed(BatchNormalization(-1))(i_model_3rd)
        i_model_4th = Activation('tanh')(i_model_4th)
        i_model_5th = Bidirectional(gru_i)(i_model_4th) 
        i_model_final = RepeatVector(self.params['CAPTION_LEN']+1)(i_model_5th) 
        #i_model = Model(inputs=i_model_input, outputs=i_model_final, name='frames_model')
        #print(i_model.summary()) 
        #tf.keras.utils.plot_model(i_model, to_file='i_model.png', show_shapes=True) 
        # _____________________________________________________________________________
        
        lstm_concatted = LSTM(1024, return_sequences=True, kernel_initializer='random_normal', recurrent_regularizer=l2(0.01))
        dense_concatted = Dense(self.params['VOCAB_SIZE'], kernel_initializer='random_normal')
        opt = RMSprop(lr=0.001, rho=0.9, epsilon=1e-8, decay=0)

        concatted = Concatenate(-1)([c_model_final, a_model_final, i_model_final])
        concatted = TimeDistributed(Dropout(0.2))(concatted)
        concatted = lstm_concatted(concatted) 
        concatted = TimeDistributed(dense_concatted)(concatted) 
        concatted = Activation('softmax')(concatted) 

        
        model = Model(inputs=[c_model_input, a_model_input, i_model_input], outputs=concatted)
        model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
        self.model = model 
        
        tf.keras.utils.plot_model(model, to_file='total.png', show_shapes=True) 
        


        # _________________________________________________________________________________



        '''
        model = Sequential()
        model.add(
            concatted
        )
        model.add(
            TimeDistributed(
                Dropout(0.2)
            )
        )
        model.add(
            LSTM(
                1024,
                return_sequences=True,
                kernel_initializer='random_normal',
                recurrent_regularizer=l2(0.01)
            )
        )
        model.add(
            TimeDistributed(
                Dense(self.params['VOCAB_SIZE'],
                      kernel_initializer='random_normal')
            )
        )
        model.add(Activation('softmax'))

        optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-8, decay=0)
        model.compile(optimizer=optimizer,
                      loss='categorical_crossentropy', metrics=['accuracy'])
        model.summary()
        log('info', ' model created!')
        self.model = model
        return model
        '''

    def get_cutoff_shape(self):
        # ResNet
        shape = (None, 2048)
        log('debug', 'model cutoff outShape: %s' % str(shape))
        return shape

    def build_cutoff_model(self):
        pass
    
    def test(self):
        x1 = tf.keras.layers.Dense(8)(np.arange(10).reshape(5, 2))
        x2 = tf.keras.layers.Dense(8)(np.arange(10, 20).reshape(5, 2))
        concatted = tf.keras.layers.Concatenate()([x1, x2])
        print(type(x1))
        print(type(x2)) 
        print(type(concatted))
        print(concatted.shape)