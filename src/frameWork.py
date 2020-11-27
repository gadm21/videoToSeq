

from utils import * 
from VideoHandler import * 
from vocab import * 
from vmodel import *


class FrameWork():

    def __init__(self, params):
        
        self.params = params 
        self.vHandler = VideoHandler(params) 
        self.vocab = Vocab(params)
        self.vmodel = VModel(parms) 


    def data_generator(self):

        while True :
            ids, frames, audios = self.vHandler.get_random_videos(self.params['BS'])
            captions_in, captions_out = self.vocab.get_captions(ids) 

            yield [[captions_in, audios, videos], captions_out]

    def train(self):
        
        dg = self.data_generator() 
        self.vmodel.model.fit(dg, steps_per_epoch=self.params['stepsPerEpoch'], epochs=self.params['epochs'])

    



if __name__ == '__main__': 
    framework = FrameWork(read_yaml()) 
    framework.train() 