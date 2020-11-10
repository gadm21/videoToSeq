
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

        c_model = Sequential()
        c_model.add(
            TimeDistributed(
                Dense(512, kernel_initializer='random_normal'),
                name='td1',
                input_shape=(self.params['CAPTION_LEN'] + 1,
                             self.params['OUTDIM_EMB'])
            )
        )
        c_model.add(
            LSTM(512, return_sequences=True, kernel_initializer='random_normal')
        )
        # c_model.summary()
        # ___________________________________________________________________________________

        a_model = Sequential()
        a_model.add(
            GRU(
                128,
                dropout=0.2,
                recurrent_dropout=0.2,
                return_sequences=True,
                input_shape=(
                    self.params['AUDIO_TimeSample'], self.params['AUDIO_n_mfcc'])
            )
        )
        a_model.add(BatchNormalization())
        a_model.add(
            GRU(
                64,
                dropout=0.2,
                recurrent_dropout=0.2,
                return_sequences=True
            )
        )
        a_model.add(BatchNormalization())
        a_model.add(Flatten())
        a_model.add(RepeatVector(self.params['CAPTION_LEN']+1))
        # a_model.summary()

        # _____________________________________________________________________________

        i_model = Sequential()
        i_model.add(
            TimeDistributed(
                Dense(1024, kernel_initializer='random_normal'),
                input_shape=self.get_cutoff_shape(),
                name='td2'
            )
        )
        i_model.add(
            TimeDistributed(
                Dropout(0.2)
            )
        )
        i_model.add(
            TimeDistributed(
                BatchNormalization(axis=-1)
            )
        )
        i_model.add(Activation('tanh'))
        i_model.add(
            Bidirectional(
                GRU(1024, return_sequences=False,
                    kernel_initializer='random_normal')
            )
        )
        i_model.add(RepeatVector(self.params['CAPTION_LEN']+1))
        # i_model.summary()

        # _________________________________________________________________________________

        print(type(c_model))
        print(type(c_model.output))
        print(type(c_model.layers[-1]))
        print(type(a_model.layers[-1]))
        print(type(i_model.layers[-1]))
        allLayers =  [c_model.layers[-1], a_model.layers[-1]]
        #allLayers =  [c_model.outputs[0], a_model.outputs[0], i_model.outputs[0]]
        #allLayers = [c_model.output, a_model.output, i_model.output]

        model = Sequential()
        model.add(
            #concatenate( [c_model.output, a_model.output, i_model.output])
            #concatenate( [c_model.outputs[0], a_model.outputs[0], i_model.outputs[0]])
            Concatenate(-1)(allLayers)
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

    def get_cutoff_shape(self):
        # ResNet
        shape = (None, 2048)
        log('debug', 'model cutoff outShape: %s' % str(shape))
        return shape

    def build_cutoff_model(self):
        pass
