

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

    def data_generator(self, BS):

        def split_sequence(seq, i):
            in_seq = seq[:i]
            in_seq += [self.vocab.padding_element]*(self.params['CAPTION_LEN']-len(in_seq))
            return in_seq
        
        
        while True : 
            
            videos, captions  = self.vHandler.get_random_videos(n = BS)
            #print("videos type:{}  captions type:{}".format(type(videos), type(captions)))
            videos = [self.vmodel.preprocess_frames(video) for video in videos] 
            videos = [self.vmodel.vid2vec(video) for video in videos] 
            
            captions = [self.vocab.caption2seq(caption) for caption in captions]
            in_vids, in_seqs, out_seqs = [], [], [] 
            


            for video, caption in zip(videos, captions):        

                for i in range(1, len(caption)):
                    in_vids.append(video) 
                    in_seq = split_sequence(caption, i)
                    out_seq = to_categorical([caption[i]], num_classes=self.params['VOCAB_SIZE'])[0]
                    in_seqs.append(in_seq)
                    out_seqs.append(out_seq)
            

            in_vids = np.asarray(in_vids).astype('float32')
            in_seqs = np.asarray(in_seqs).astype('float32')
            out_seqs = np.asarray(out_seqs).astype('float32') 

            #print("{}_{}_{}".format(type(in_vids), type(in_seqs), type(out_seq)))
            #print("{}_{}_{}".format(in_vids.shape, in_seqs.shape, out_seqs.shape))

            yield ([in_seqs, in_vids], out_seqs)
            
    def train(self):
        print("training...")
        dg = self.data_generator(self.params['BS'])
        self.vmodel.model.fit(dg, steps_per_epoch=self.params['stepsPerEpoch'], epochs=self.params['epochs'])
        
        print("ending training, peacfully...")
    
    def predict(self, video = None, gt_caps = None):
        if not video :
            video, gt_caps = self.vHandler.get_random_videos(n=1)
        
        if gt_caps:
            print("ground truth caption:", gt_caps[0]) 

        preprocessed_video = self.vmodel.preprocess_frames(video)
        videoVec = self.vmodel.vid2vec( preprocessed_video[0])
        
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
    framework = FrameWork(read_yaml()) 
    framework.predict() 
