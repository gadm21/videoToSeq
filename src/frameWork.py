

from utils import * 
from VideoHandler import * 
from vocab import * 
from vmodel import *


class FrameWork():

    def __init__(self, params):
        
        self.params = params 
        self.vHandler = VideoHandler(params) 
        self.vocab = Vocab(params)
        self.vmodel = VModel(params) 


    def data_generator(self):

        while True :
            videos, captions = self.vHandler.get_random_videos(self.params['BS'])
            #captions_in, captions_out = self.vocab.get_captions(ids) 

            yield [[captions_in, audios, videos], captions_out]

    def train(self):
        
        dg = self.data_generator() 
        self.vmodel.model.fit(dg, steps_per_epoch=self.params['stepsPerEpoch'], epochs=self.params['epochs'])

    

    def dev_train(self):

        def split_sequence(seq, i):
            in_seq, out_seq = seq[:i], seq[i:] 

        videos, captions  = self.vHandler.get_random_videos(n = self.params['BS']) 
        videos = [self.vmodel.preprocess_partial_model(video) for video in videos]
        captions = [self.vocab.caption2seq(caption) for caption in captions]
        
        batch = 0
        in_vids = [] 
        in_seqs = [] 
        out_seqs = [] 

        for video, caption in zip(videos, captions):
            for i in range(len(caption)):
                in_vids.append(video) 
                in_seqs.append(caption[:i])
                out_seqs.append(caption[i:])
            batch += 1
            if batch >= self.params['BS'] : 
                in_vids = np.array(in_vids) 
                in_seqs = np.array(in_seqs)
                out_seqs = np.array(out_seqs) 
                print(in_vids.shape)
                print(in_seqs.shape) 
                print(out_seqs.shape)
                #return in_vids, in_seqs, out_seqs



        
if __name__ == '__main__': 
    framework = FrameWork(read_yaml()) 
    framework.dev_train() 
