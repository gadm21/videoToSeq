
from utils import *
from VideoHandler import VideoHandler
from vocab import Vocab

#import matplotlib.pyplot as plt
import numpy as np
import os
import sys


from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.layers import Dropout, Flatten, RepeatVector, Activation, Concatenate
from tensorflow.keras.layers import Embedding, Conv2D, MaxPooling2D, LSTM, GRU, BatchNormalization
from tensorflow.keras.layers import TimeDistributed, Dense, Input, GlobalAveragePooling2D, Bidirectional
from tensorflow.keras.layers import LayerNormalization
from tensorflow.keras.layers.experimental.preprocessing import Rescaling

from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.optimizers.schedules import PolynomialDecay
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.preprocessing import image
from tensorflow.keras.regularizers import l2


import tensorflow as tf
import tensorflow.keras.backend as K


tf.get_logger().setLevel('INFO')

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



class MultiHeadSelfAttention(tf.keras.layers.Layer):

    def __init__(self, embed_dim, num_heads = 2):
        super(MultiHeadSelfAttention, self).__init__()  
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        if embed_dim % num_heads != 0 : raise ValueError("embed_dim should be divisible by num_heads")

        self.projection_dim = embed_dim // num_heads 
        self.query_dense = Dense(embed_dim) 
        self.key_dense = Dense(embed_dim)
        self.value_dense = Dense(embed_dim) 
        self.combine_heads = Dense(embed_dim)

    def attention(self, query, key, value):
        score = tf.matmul(query, key, transpose_b=True)
        dim_key = tf.cast(tf.shape(key)[-1], tf.float32) 
        scaled_score = score / tf.math.sqrt(dim_key) 
        weights = tf.nn.softmax(scaled_score, axis=-1)
        output = tf.matmul(weights, value) 
        return output, weights 
    
    def separate_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.projection_dim))
        return tf.transpose(x, perm=[0,2,1,3])
    
    def call(self, inputs):
        batch_size = tf.shape(inputs)[0] 

        query = self.query_dense(inputs) 
        key = self.key_dense(inputs)
        value = self.value_dense(inputs) 

        query = self.separate_heads(query, batch_size)
        key = self.separate_heads(key, batch_size)
        value = self.separate_heads(value, batch_size)

        attention, weights = self.attention(query, key, value) 
        attention = tf.transpose(attention, perm=[0,2,1,3])
        concat_attention = tf.reshape(attention, (batch_size, -1, self.embed_dim))

        output = self.combine_heads(concat_attention) 
        return output 


class TransformerBlock(tf.keras.layers.Layer):

    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1):
        super(TransformerBlock, self).__init__()
        self.att = MultiHeadSelfAttention(embed_dim, num_heads) 
        self.ffn = Sequential([Dense(ff_dim, activation='relu'), Dense(embed_dim)])
        self.norm_layer1 = LayerNormalization(epsilon=1e-6)
        self.norm_layer2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(dropout) 
        self.dropout2 = Dropout(dropout) 

    def call(self, inputs, training):

        attn_output = self.att(inputs) 
        attn_output = self.dropout1(attn_output, training = training) 
        out1 = self.norm_layer1(inputs+attn_output)

        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training = training) 
        out2 = self.norm_layer2(out1 + ffn_output) 

        return out2



class VisionTransformer(tf.keras.Model):

    def __init__(self, image_size, patch_size, num_layers, num_classes,
                d_model, num_heads, mlp_dim, channels= 3, dropout= 0.1) : 

        super(VisionTransformer, self).__init__()
        
        self.patch_size = patch_size 
        self.d_model = d_model
        self.num_layers = num_layers

        num_patches = (image_size // patch_size)**2 #number of grid cells 
        self.patch_dim = channels * (patch_size ** 2 )

        self.rescale = Rescaling(1. / 255) 
        self.pos_emb = self.add_weight('pos_emb', shape=(1, num_patches+1, d_model))
        self.class_emb = self.add_weight('class_emb', shape=(1,1,d_model))
        self.patch_proj = Dense(d_model)
        self.enc_layer = [TransformerBlock(d_model, num_heads, mlp_dim, dropout) for _ in range(num_layers)]
        self.mlp_layer = Sequential([
            Dense(mlp_dim, activation='relu'),
            Dropout(dropout),
            Dense(num_classes)
        ])

    def extract_patches(self, images):
        batch_size = tf.shape(images)[0] 
        patches = tf.image.extract_patches(
            images = images,
            sizes = [1, self.patch_size, self.patch_size, 1],
            strides = [1, self.patch_size, self.patch_size, 1],
            rates = [1,1,1,1],
            padding = 'VALID'
        )

        patches = tf.reshape(patches, [batch_size, -1, self.patch_dim])
        return patches
    
    def call(self, x, training):
        batch_size = tf.shape(x)[0]
        x = self.rescale(x)
        patches = self.extract_patches(x)
        x = self.patch_proj(patches) 

        class_emb = tf.broadcast_to(self.class_emb, [batch_size, 1, self.d_model])
        x = tf.concat([class_emb, x], axis=1) 
        x += self.pos_emb

        for layer in self.enc_layer : x = layer(x, training)
        
        x = self.mlp_layer(x[:, 0])
        return x



def train_vision_transformer():


    model = VisionTransformer(image_size=150, patch_size= 50, num_layers = 2, num_classes = 10,
                             d_model = 300, num_heads = 2, mlp_dim = 100)
    
    model.compile(
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        optimizer = tf.keras.optimizers.Adam(lr=0.001, decay= 1e-4),
        metrics = ['accuracy'],
    )

    import tensorflow_datasets as tfds

    ds = tfds.load('cifar10', as_supervised = True) 
    ds_train = (ds['train'].batch(2))
    
    model.fit(ds_train, epochs = 3)

    print("done")
    model.summary() 
    print("dond")

















class VModel:

    def __init__(self, params):

        K.clear_session()

        self.callbacks = []
        self.params = params
        self.model_path = params['model_path']

        self.build_cnn_model()
        self.build_model()
        self.compile_model()
    


    def preprocess_frames(self, video):
        
        video = [self.preprocessing_func(frame) for frame in list(video)]
        video = np.array(video, dtype= np.float32)

        return video
        
    def vid2vec(self, video):
        return self.cnn_model.predict(video) 

    def build_cnn_model(self):
        self.preprocessing_func = preprocess_input
        
        self.cnn_model = InceptionV3(input_shape = (self.params['FRAME_SIZE'], self.params['FRAME_SIZE'], 3), include_top = False, pooling = 'avg') 
    
        for layer in self.cnn_model.layers:
            layer.trainable = False 
 
        #print(self.cnn_model.summary())


    def build_model(self):

        #_________________________________________________________
        #_______________custom layers____________________________
        conv2d_1 = Conv2D(5, 7, dilation_rate=3, activation='relu')
        conv2d_2 = Conv2D(7, 5, dilation_rate=2, activation='tanh')
        conv2d_3 = Conv2D(10, 3, dilation_rate=2, activation='relu')
        conv2d_4 = Conv2D(15, 3,  activation='tanh')
        dense_i = Dense(1024, kernel_initializer='random_normal') 
        gru_i = GRU(1024, return_sequences=False, kernel_initializer='random_normal') 

        dense_1 = Dense(200, activation='relu')
        #_________________________________________________________
        #_________________________________________________________


        #________________c_model layers_________________________
        input_1 = Input(shape = (self.params['CAPTION_LEN']))
        embedding_1 = Embedding(self.params['VOCAB_SIZE'], self.params['OUTDIM_EMB'])(input_1) 
        lstm_1 = LSTM(150, return_sequences=True)(embedding_1) 
        time_dist_1 = TimeDistributed(Dense(100, activation='relu'))(lstm_1) 
        
        #_______________v_model layers____________________________

        input_2 = Input(shape=(self.params['FRAMES_LIMIT'], self.params['VIDEO_VEC'])) 
        time_dist_2 = TimeDistributed(Dense(1024, activation='relu'))(input_2)
        time_dist_2 = TimeDistributed(Dropout(0.2))(time_dist_2) 
        lstm_2 = LSTM(500)(time_dist_2)
        rep_vec = RepeatVector(self.params['CAPTION_LEN'])(lstm_2)  


        #_______________concattenated layers_____________________
        concatted = Concatenate(2)([time_dist_1, rep_vec]) 
        flattened = LSTM(100)(concatted) 
        final = Dense(self.params['VOCAB_SIZE'], activation='softmax')(flattened) 

        self.model = Model(inputs = [input_1, input_2], outputs = final) 
        
        self.plot_model()
        print("model built")


    def compile_model(self):
        self.model.compile(
            loss= 'categorical_crossentropy',
            optimizer= RMSprop(lr = self.params['learning_rate'], epsilon = 1e-8, rho = 0.9),
            metrics=["accuracy"]
        )
    
    def plot_model(self):
        tf.keras.utils.plot_model(self.model, 'visuals/light_model.png', show_shapes=True, show_layer_names=False) 
        tf.keras.utils.plot_model(self.model, 'visuals/specific_light_model.png', show_shapes=True) 




if __name__ == '__main__':
    params = read_yaml() 
    vmodel = VModel(params)
    
    sample = np.ones((20, params['FRAME_SIZE'],params['FRAME_SIZE'],3), dtype = np.float32)
    output = vmodel.vid2vec(vmodel.preprocess_frames(sample)) 

    output_vid = np.array([output], dtype = np.float32)
    output_seq = np.array([np.ones(params['CAPTION_LEN'], dtype = np.float32)])
    output = [output_seq, output_vid]
    print(sample.shape, " ", output_vid.shape, " ", output_seq.shape)
    
    output2 = vmodel.model.predict(output)

    print(output2.shape)


