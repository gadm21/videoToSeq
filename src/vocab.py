

from utils import *
from videoHandler import * 
import os
import numpy as np
import pickle

#TODO rewrite vocab to include the embeddings and vocab from the 
#raw data not from downloaded videos so that it is not limited
#to the captions of the downloaded videos

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

        self.build_vocab() 
        self.load_embeddings()


    

    def load_embeddings(self):
        
        if os.path.exists(self.params['embeddings_file']):
            with open(self.params['embeddings_file'], 'rb') as f:
                self.embeddings = pickle.load(f) 
            log('info', 'embeddings loaded') 
        else:
            self.embeddings = dict()
            
            for token in self.vocab :
                if token not in self.embeddings.keys() :
                    self.embeddings[token] = get_embeddings(n=self.params['OUTDIM_EMB'])
            log('info', 'embeddings created')  

            self.save_embeddings() 
            log('info', 'embeddings saved') 

    def save_embeddings(self):
        with open(self.params['embeddings_file'], 'wb') as f :
            pickle.dump(self.embeddings, f) 

    def build_vocab(self):
        if os.path.exists(self.params['word2ix_file']):
            with open(self.params['word2ix_file'], 'rb') as f:
                self.word2ix = pickle.load(f)

            self.ix2word = {index:word for word,index in self.word2ix.items()} 
            self.vocab = [word for word in self.word2ix.keys()] 
            log('info', 'word2ix loaded')

        else:
            
            self.vocab = dict() 
            for caption in self.captions :
                for sentence in caption : 
                    for token in tokenize_caption(sentence):
                        self.vocab[token] = self.vocab.get(token, 0) + 1 
            
            self.vocab = [pair[0] for pair in 
                sorted(self.vocab.items(), key = lambda item : item[1], reverse= True)
                [: self.params['VOCAB_SIZE']-len(self.specialWords)]]
            self.vocab.extend(self.specialWords)

            self.word2ix = {word:index for index,word in enumerate(self.vocab)} 
            self.ix2word = {index:word for index,word in enumerate(self.vocab)}

            log('info', 'vocab built')

            with open(self.params['word2ix_file'], 'wb') as f :
                pickle.dump(self.word2ix, f) 

            log('info', 'vocab saved')




if __name__ == "__main__":
    params = read_yaml()
    vh = videoHandler(params) 
    v = Vocab(params,vh.downloaded)
