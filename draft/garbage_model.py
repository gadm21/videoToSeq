     




     
    def build_cutoff_model(self):
        base = ResNet50(include_top=False, weights='imagenet', pooling='avg') 
        for layer in base.layers : 
            layer.trainable = False 
        self.co_model = Model(base.input, base.output) 

    def preprocess_partial_model(self, frames):
        return self.co_model.predict(preprocess_input(frames))
    




    def build_model(self):

        '''
        #log('debug', "creating model (CNN cutoff) with Vocab size: %d" % self.params['VOCAB_SIZE'])
        print("building model...")
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
        conv2d_1 = Conv2D(5, 10, dilation_rate=3, activation='relu')
        pool2d = MaxPooling2D(2, 2) 
        conv2d_2 = Conv2D(10, 7, dilation_rate=3, activation='tanh')
        conv2d_3 = Conv2D(15, 5, dilation_rate=2, activation='relu')
        conv2d_4 = Conv2D(20, 5, dilation_rate=2, activation='tanh')
        conv2d_5 = Conv2D(30, 3, activation='tanh')

        dense_i = Dense(1024, kernel_initializer='random_normal') 
        gru_i = GRU(1024, return_sequences=False, kernel_initializer='random_normal') 

        v_model_input = Input(shape = ( self.params['FRAME_SIZE'], self.params['FRAME_SIZE'], 1), name='frame')

        v_model = conv2d_1(pool2d(v_model_input)) 
        v_model = conv2d_2(pool2d(v_model)) 


        '''


        v_model_3 = conv2d_3(v_model_2p) 
        v_model_3p = pool2d(v_model_3) 
        
        #v_model = TimeDistributed(conv2d_4)(v_model) 
        #v_model = TimeDistributed(conv2d_5)(v_model)
        '''

        '''
        v_model = TimeDistributed(Dropout(0.2))(v_model) 
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
        #self.callbacks.append(LearningRateScheduler(schedule))
        self.callbacks.append(model_checkpoint_callback)
        opt = RMSprop(lr=0.0001, rho=0.9, epsilon=1e-8, decay=0.5)
        
        model = Model(inputs=[c_model_input, i_model_input], outputs=concatted)
        model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
        #tf.keras.utils.plot_model(model, to_file='total.png', show_shapes=True) 
        '''

        opt = RMSprop(lr=0.0001, rho=0.9, epsilon=1e-8, decay=0.5)
        model = Model(inputs=v_model_input, outputs=v_model)
        model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
        tf.keras.utils.plot_model(model, to_file='test.png', show_shapes=True) 
        
        model.summary()
        self.model = model 