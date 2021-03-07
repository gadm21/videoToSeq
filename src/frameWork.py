

import os
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"]="true"

from utils import * 
from VideoHandler import * 
from vocab import * 
from vmodel import *
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.utils import Sequence

from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler, ModelCheckpoint




'''

    def data_generator(self):

        def split_sequence(seq, i):
            in_seq = seq[:i]
            in_seq += [self.vocab.padding_element]*(self.params['CAPTION_LEN']-len(in_seq))
            return in_seq
        
        BS = self.params['BS']
        while True : 
            
            ids = np.random.choice([id[:-4] for id in os.listdir(self.params['vids_dir'])], BS)
            ids =[id[:-4] for id in self.id_list[self.id_counter: self.id_counter + BS]]
            self.id_counter += len(ids)
            log('info', 'selected_ids:{}'.format(ids))
            log('info', 'id counter:{}'.format(self.id_counter))
            log('info', '\n\n')


            videos = [self.vHandler.get_video_by_id(id) for id in ids] 
            captions = [self.vocab.get_caption_by_id(id) for id in ids]
            log('info', 'caption length:{}'.format(len(captions)) )
            log('info', 'captions:\n')
            for caption in captions:
                log('info', '\t {}'.format(caption))
            log('info', '\n\n')

            videos = [self.vmodel.preprocess_frames(video) for video in videos] 
            videos = [self.vmodel.vid2vec(video) for video in videos] 
            
            captions = [self.vocab.caption2seq(caption) for caption in captions]
            log('info', 'captions:')
            for caption in captions:
                log('info', '\t {}'.format(caption))
            log('info', '\n\n')

            in_vids, in_seqs, out_seqs = [], [], [] 
            
            log('info', 'looping over batch:')
            for video, caption in zip(videos, captions):        
                
                log('info', 'target caption:{}'.format(caption))
                for i in range(1, len(caption)):
                    in_vids.append(video) 
                    in_seq = split_sequence(caption, i)
                    out_seq = to_categorical([caption[i]], num_classes=self.params['VOCAB_SIZE'])[0]
                    in_seqs.append(in_seq)
                    out_seqs.append(out_seq)
                    log('info', 'in sequence:{}'.format(in_seq))
                    log('info', 'out sequence:{}'.format(out_seq))
                    log('info', '\n')
            log('info', 'ending loop over batch')
            

            in_vids = np.asarray(in_vids).astype('float32')
            in_seqs = np.asarray(in_seqs).astype('float32')
            out_seqs = np.asarray(out_seqs).astype('float32') 

            log('info', '  size of:: in_vids:{} in_seqs:{} out_seqs:{}'.format(in_vids.shape, in_seqs.shape, out_seqs.shape))
            log('info', '\n\n\n')

            yield ([in_seqs, in_vids], out_seqs)






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

        lstm_1 = LSTM(100, return_sequences=True)(time_dist_1) 
        time_dist_1 = TimeDistributed(Dense(50, activation='relu'))(lstm_1) 
        #time_dist_1 = TimeDistributed(Dropout(0.3))(time_dist_1) 
        #_______________v_model layers____________________________

        input_2 = Input(shape=(self.params['FRAMES_LIMIT'], self.params['VIDEO_VEC'])) 
        time_dist_2 = TimeDistributed(Dense(1024, activation='relu'))(input_2)
        #time_dist_2 = TimeDistributed(Dropout(0.3))(time_dist_2) 
        lstm_2 = LSTM(1024, return_sequences = False)(time_dist_2)
        lstm_2 = Dense(500, activation = 'relu')(lstm_2)
        rep_vec = RepeatVector(self.params['CAPTION_LEN'])(lstm_2)  


        #_______________concattenated layers_____________________
        concatted = Concatenate(2)([time_dist_1, rep_vec]) 
        concatted = LSTM(300, return_sequences= False)(concatted)
        flattened = Dense(200, activation = 'tanh')(concatted) 
        final = Dense(self.params['VOCAB_SIZE'], activation='softmax')(flattened) 

        self.model = Model(inputs = [input_1, input_2], outputs = final) 
        
        self.plot_model()
        print("model built")

'''




class DataGenerator(Sequence):

    def __init__(self, params, vocab, vHandler, vmodel):
        
        self.params = params
        self.vHandler = vHandler
        self.vocab = vocab
        self.vmodel = vmodel

        self.id_list = os.listdir(self.params['vids_dir'])
        self.BS = self.params['BS']
        self.id_counter = 0
    
    def __len__(self):
        the_length =  len(self.id_list) // self.BS
        log('info', 'the length:{}'.format(the_length))
        return the_length

    def __getitem__(self, idx):

        def split_sequence(seq, i):
            in_seq = seq[:i]
            in_seq += [self.vocab.padding_element]*(self.params['CAPTION_LEN']-len(in_seq))
            return in_seq


        ids = [id[:-4] for id in self.id_list[idx*self.BS : (idx+1)*self.BS]]
        self.id_counter += len(ids)
        log('info', 'selected_ids:{}'.format(ids))
        log('info', 'id counter:{}'.format(self.id_counter))
        log('info', '\n\n')


        videos = [self.vHandler.get_video_by_id(id) for id in ids] 
        captions = [self.vocab.get_caption_by_id(id) for id in ids]
        log('info', 'caption length:{}'.format(len(captions)) )
        log('info', 'captions:\n')
        for caption in captions:
            log('info', '\t {}'.format(caption))
        log('info', '\n\n')

        videos = [self.vmodel.preprocess_frames(video) for video in videos] 
        videos = [self.vmodel.vid2vec(video) for video in videos] 
        
        captions = [self.vocab.caption2seq(caption) for caption in captions]
        log('info', 'captions:')
        for caption in captions:
            log('info', '\t {}'.format(caption))
        log('info', '\n\n')

        in_vids, in_seqs, out_seqs = [], [], [] 
        
        log('info', 'looping over batch:')
        for video, caption in zip(videos, captions):        
            
            log('info', 'target caption:{}'.format(caption))
            for i in range(1, len(caption)):
                in_vids.append(video) 
                in_seq = split_sequence(caption, i)
                out_seq = to_categorical([caption[i]], num_classes=self.params['VOCAB_SIZE'])[0]
                in_seqs.append(in_seq)
                out_seqs.append(out_seq)
                log('info', 'in sequence:{}'.format(in_seq))
                log('info', 'out sequence:{}'.format(out_seq))
                log('info', '\n')
        log('info', 'ending loop over batch')
        

        in_vids = np.asarray(in_vids).astype('float32')
        in_seqs = np.asarray(in_seqs).astype('float32')
        out_seqs = np.asarray(out_seqs).astype('float32') 

        log('info', '  size of:: in_vids:{} in_seqs:{} out_seqs:{}'.format(in_vids.shape, in_seqs.shape, out_seqs.shape))
        log('info', '\n\n\n')

        return ([in_seqs, in_vids], out_seqs)        











class FrameWork():

    def __init__(self, params):
        
        self.params = params 
        self.vHandler = VideoHandler(params) 
        self.vocab = Vocab(params)
        self.vmodel = VModel(params) 
        self.id_counter = 0
        self.id_list = os.listdir(self.params['vids_dir'])
        log('info', 'num videos:{} video ids:{}'.format(len(self.id_list), self.id_list))


    def prepare_callbacks(self):

        def lr_scheduler(epoch, lr):
            rounds = epoch // 20
            new_lr = lr
            for _ in range(rounds) :
                new_lr /= 5
            return new_lr

        model_path = self.params['model_path']

        modelCheckpoint_callback = ModelCheckpoint( model_path, monitor='loss', save_best_only=True, save_weights_only=True)

        lrScheduler_callback = LearningRateScheduler(lr_scheduler)

        earlyStopping_callback = EarlyStopping( monitor= 'loss', patience=13)

        return [] 
        return [modelCheckpoint_callback, earlyStopping_callback]


    
    def train(self):
        
        dg = DataGenerator(self.params, self.vocab, self.vHandler, self.vmodel)
        log('info', 'training...')

        callbacks = self.prepare_callbacks()
        self.vmodel.model.fit(dg, epochs=self.params['epochs'], callbacks=callbacks)
        
        log('info', 'ending training peacefully...')
    


    def predict(self, video = None, gt_caps = None, n=1):
        if video is None :
            video, gt_caps =  self.vHandler.get_random_videos(n=n), ''
        
        if gt_caps:
            print("ground truth caption:", gt_caps[0]) 

        preprocessed_video = self.vmodel.preprocess_frames(video)
        videoVec = self.vmodel.vid2vec( preprocessed_video[0]) # choose only the first video
        
        caption_len = self.params['CAPTION_LEN']
        current_len = 0
        caption = ['seq_start']

        while current_len < caption_len:
            
            padded_caption = self.vocab.pad(caption.copy())
            in_seq_cap = np.array([self.vocab.word2ix.get(word, self.vocab.word2ix['seq_unkown']) for word in padded_caption], dtype = np.float32)
            in_seq_cap = np.array([in_seq_cap]).astype(np.float32)
            in_seq_vid = np.array([videoVec]).astype(np.float32)
            
            inputt = [in_seq_cap, in_seq_vid]
            verbose_seq = [self.vocab.ix2word[word] for word in in_seq_cap[0].tolist()]
            #print(verbose_seq)

            pred = self.vmodel.model.predict(inputt)[0] 
            ix = np.argmax(pred) 
            word = self.vocab.ix2word.get(ix, 'seq_unkown')
            caption.append(word) 
            if word=='seq_end' : break

            current_len += 1

        predicted = ' '.join(caption)
        return predicted
        

    def dev_train(self):

        print("vocab::", len(self.vocab.vocab))
        def split_sequence(seq, i):
            in_seq = seq[:i]
            in_seq += [self.vocab.padding_element]*(self.params['CAPTION_LEN']-len(in_seq))
            return in_seq
        


        while True : 
            videos, captions  = self.vHandler.get_random_videos(n = self.params['BS'])
            videos = [self.vmodel.preprocess_partial_model(video) for video in videos] 
            captions = [self.vocab.caption2seq(caption) for caption in captions]
            in_vids, in_seqs, out_seqs = [], [], [] 
            

            for video, caption in zip(videos, captions):        

                for i in range(1, len(caption)):
                    in_vids.append(video) 
                    in_seq = split_sequence(caption, i)
                    out_seq = to_categorical([caption[i]], num_classes=self.params['VOCAB_SIZE'])
                    in_seqs.append(in_seq)
                    out_seqs.append(out_seq)
                    
            in_vids = np.array(in_vids) 
            in_seqs = np.array(in_seqs)
            out_seqs = np.array(out_seqs) 
            print(in_vids.shape)
            print(in_seqs.shape) 
            print(out_seqs.shape)
            print()
            print()
            return in_vids, in_seqs, out_seqs 
            




if __name__ == '__main__': 
    params = read_yaml()
    framework = FrameWork(params) 

    vids_dir = framework.vHandler.vids_dir 
    ids = [id[:-4] for id in os.listdir(vids_dir)]

    train = params['train']

    if train :
        framework.train()
    else:
        for id in ids :
            pred = framework.predict(framework.vHandler.get_video_by_id(id)) 
            print("predicted:", pred)
            