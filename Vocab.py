

from the_utils import *
from fake_embeddings import create_embeddings
from VideoDataHandler import * 
import os
import numpy as np
import pickle


class Vocab:

    def __init__(self, params, downloaded):

        self.params = params 
        self.videos = [info[0] for info in downloaded]
        self.captions = [info[1:] for info in downloaded] 

        self.specialWords = dict()
        self.specialWords['START'] = '>'
        self.specialWords['END'] = '<'
        self.specialWords['NONE'] = '?!?'
        self.specialWords['EXTRA'] = '___'

        self.load_embeddings()
        self.build_vocab() 

        log('info', 'Vocab built!')

    

    def load_embeddings(self):
        
        if os.path.exists(self.params['embeddings_file']):
            log('info', 'embeddings_file exists')
            with open(self.params['embeddings_file'], 'rb') as f:
                self.embeddings = pickle.load(f) 
            log('info', 'embeddings loaded')
            return False
        else:
            self.embeddings = dict()
            for caption in self.captions:
                for sentence in caption:
                    tokens = sentence.split(' ')
                    for token in tokens :
                        if token not in self.embeddings.keys() :
                            self.embeddings[token] = create_embeddings(n=self.params['OUTDIM_EMB'])
            log('info', 'embeddings created') 
            log('info', 'embeddings size:{}'.format(len(self.embeddings.keys())))
            log('info', 'embeddings size:{}'.format(self.embeddings['car'].shape))
            self.save_embeddings() 

    def save_embeddings(self):
        with open(self.params['embeddings_file'], 'wb') as f :
            pickle.dump(self.embeddings, f) 
            log('info', 'embeddings saved')

    def build_vocab(self):
        if os.path.exists(self.params['vocab_file']):
            log('info', 'vocab_file exists') 
            with open(self.params['vocab_file'], 'rb') as f:
                self.ind2word = pickle.load(f)
            log('info', 'vocab_file loaded')
        else:
            log('info', 'building vocab (ind2word)')

            all_words = [word for word in self.embeddings.keys()]
            all_words.extend(list(self.specialWords.values()))

            print("len:", len(all_words)) 
            print("words:", all_words[:5])




if __name__ == "__main__":
    params = read_yaml()
    vh = VideoDataHandler(params) 
    v = Vocab(params,vh.downloaded)
