

from utils import *
from VideoHandler import *
import os
import numpy as np
import pickle

import nltk
from textblob import TextBlob, Word 
from collections import Counter
from scipy.spatial import distance

'''

    def load_embeddings(self):

        if os.path.exists(self.params['embeddings_file']):
            with open(self.params['embeddings_file'], 'rb') as f:
                self.embeddings = pickle.load(f)
            log('info', 'embeddings loaded')
        else:
            self.embeddings = dict()

            for token in self.vocab:
                if token not in self.embeddings.keys():
                    self.embeddings[token] = get_embeddings(
                        n=self.params['OUTDIM_EMB'])
            #log('info', 'embeddings created')

            # self.save_embeddings()
            #log('info', 'embeddings saved')

    def save_embeddings(self):
        with open(self.params['embeddings_file'], 'wb') as f:
            pickle.dump(self.embeddings, f)

    def build_vocab(self):
        if os.path.exists(self.params['word2ix_file']) and False :
            with open(self.params['word2ix_file'], 'rb') as f: self.word2ix = pickle.load(f)
            self.ix2word = {index: word for word, index in self.word2ix.items()}
            self.vocab = [word for word in self.word2ix.keys()]
            print("{} vocab loaded".format(len(self.vocab)))

        else:

            self.vocab = dict()
            for caption in self.captions:
                for token in tokenize_caption(caption['caption']):
                    self.vocab[token] = self.vocab.get(token, 0) + 1

            self.vocab = [pair[0] for pair in
                          sorted(self.vocab.items(),
                                 key=lambda item: item[1], reverse=True)
                          [: self.params['VOCAB_SIZE']-len(self.specialWords)]]
            self.vocab.extend(self.specialWords)

            self.word2ix = {word: index for index,
                            word in enumerate(self.vocab)}
            self.ix2word = {index: word for index,
                            word in enumerate(self.vocab)}
            self.padding_element = self.word2ix['seq_extra']
            print("vocab built")

            with open(self.params['word2ix_file'], 'wb') as f:
                pickle.dump(self.word2ix, f)



'''


class Vocab:

    def __init__(self, params):

        self.params = params
        self.raw_data = read_json(params['training_data'])
        self.captions = self.raw_data['sentences']
        self.videos = self.raw_data['videos']

        self.full_captions = [] 
        self.svo_captions = [] 

        self.specialWords = ['seq_start', 'seq_end', 'seq_unkown', 'seq_extra']
        self.padding_element = None

        #self.vocab = set()
        self.glove = self.load_glove()
        self.vid2cap = dict() 

        self.create_vid2cap() 
        self.build_vocab()
        #self.load_embeddings()


    def create_vid2cap(self):

        video_ids = [id[:-4] for id in os.listdir(self.params['vids_dir'])]
        for video_id in video_ids : self.vid2cap[video_id] = [] 
        
        for sentence in self.captions :
            if sentence['video_id'] not in self.vid2cap: continue
            video_id = sentence['video_id']
            caption = sentence['caption'] 
            tokens = tokenize_caption(caption) 
            words = [Word(word).lemmatize(myPosTagger(pos)) for (word, pos) in nltk.pos_tag(tokens) if pos[:2]=='NN' or pos[:2]=='VB']
            if len(words) != 3 : continue

            self.full_captions.append((caption, video_id))
            caption = ' '.join(words)
            self.svo_captions.append((caption, video_id)) 
            self.vid2cap[video_id].append(caption)


    def load_glove(self):
        with open(self.params['glove_file'], 'rb') as f:
            return pickle.load(f)
            

    def build_vocab(self):
        
        self.vocab = dict()

        for caption, video_id in self.svo_captions:
            for token in tokenize_caption(caption):
                self.vocab[token] = self.vocab.get(token, 0) + 1

        self.vocab = {pair[0]:pair[1] for pair in
                        sorted(self.vocab.items(), key=lambda item: item[1], reverse=True)}
                                
        for word in self.specialWords :
            self.vocab[word] = 1
            
        #self.reduce_vocab()
        vocabList = list(self.vocab.keys())
        
        self.word2ix = {word: index for index, word in enumerate(vocabList)}
        self.ix2word = {index: word for index, word in enumerate(vocabList)}
        self.padding_element = self.word2ix['seq_extra']
        print("vocab built")

        with open(self.params['vocab_file'], 'wb') as f: pickle.dump(self.vocab, f)
        with open(self.params['ix2word_file'], 'wb') as f: pickle.dump(self.ix2word, f)
        with open(self.params['ix2word_file'], 'wb') as f: pickle.dump(self.ix2word, f)
        print("vocab saved")

    '''
    def reduce_vocab(self, threshold = 1.3, distance_func = distance.euclidean):

        def get_distance(word1, word2, glove, distance_func, silent_kill = True):
            u = self.glove.get(word1, None) 
            v = self.glove.get(word2, None) 

            if u is None or v is None : 
                if not silent_kill :  print("{} or {} dosnot exist in GloVe".format(word1, word2)) 
                return 0
                        
            return distance_func(u, v)

        
        map_vocab = dict() 
        vocabList = list(vocab.values)
        for word in vocabList :

        distances = [get_distance(reference, word, glove, distance.euclidean, silent_kill=True) for word in plain_nouns]
        distances = np.array(distances)

        threshold = distances.mean() - (distances.std()*1.2)
        idx = np.where((distances < threshold) & ( distances != 0))
        close_words = [plain_nouns[index] for index in idx[0]]
        close_words.append(reference) 
    '''

    def get_caption_by_id(self, video_id, max_captions=1):
        #return self.vid2cap[video_id][1:max_captions+1]
        return self.vid2cap[video_id][1]

    def caption2seq(self, caption):
        if isinstance(caption, str):
            caption = tokenize_caption(caption)
        caption = caption[:self.params['CAPTION_LEN'] - 2]
        caption = ['seq_start'] + caption + ['seq_end']
        #print("caption c2s:", caption) 

        return [self.word2ix.get(word, self.word2ix['seq_unkown']) for word in caption]  


    def pad(self, seq):
        seq += [self.padding_element] * max(0, self.params['CAPTION_LEN']-len(seq))
        return seq


if __name__ == "__main__":
    params = read_yaml()
    v = Vocab(params)
    #video_ids = ['video8331', 'video231', 'video2752', 'video226']

    sentence = 'seq_start teacher write chalkboard seq_end seq_extra'
    for word in tokenize_caption(sentence) :
        print(v.word2ix[word])

    
