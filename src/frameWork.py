

import os
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"]="true"

from utils import * 
from VideoHandler import * 
from vocab import * 
from vmodel import *
from tensorflow.keras.utils import to_categorical

class FrameWork():

    def __init__(self, params):
        
        self.params = params 
        self.vHandler = VideoHandler(params) 
        self.vocab = Vocab(params)
        self.vmodel = VModel(params) 

    def data_generator(self):

        def split_sequence(seq, i):
            in_seq = seq[:i]
            in_seq += [self.vocab.padding_element]*(self.params['CAPTION_LEN']-len(in_seq))
            return in_seq
        

        while True : 
            
            videos, captions  = self.vHandler.get_random_videos(n = self.params['BS'])
            videos = [self.vmodel.preprocess_frames(video) for video in videos] 
            captions = [self.vocab.caption2seq(caption) for caption in captions]
            in_vids, in_seqs, out_seqs = [], [], [] 
            
            for video, caption in zip(videos, captions):        

                for i in range(1, len(caption)):
                    in_vids.append(video) 
                    in_seq = split_sequence(caption, i)
                    out_seq = to_categorical([caption[i]], num_classes=self.params['VOCAB_SIZE'])[0]
                    in_seqs.append(in_seq)
                    out_seqs.append(out_seq)
                    
            in_vids = np.asarray(in_vids) 
            in_seqs = np.asarray(in_seqs)
            out_seqs = np.asarray(out_seqs) 
            
            yield ([in_seqs, in_vids], out_seqs)
            
    def train(self):
        print("training...")
        dg = self.data_generator() 
        self.vmodel.model.fit(dg, steps_per_epoch=self.params['stepsPerEpoch'], epochs=self.params['epochs'], callbacks = self.vmodel.callbacks)
        print("ending training, peacfully...")
    

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
    framework = FrameWork(read_yaml()) 
    framework.train() 
