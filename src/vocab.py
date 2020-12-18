

from utils import *
from VideoHandler import *
import os
import numpy as np
import pickle


'''
TODO get random caption from a video id
TODO rewrite vocab to include the embeddings and vocab from the 
    raw data not from downloaded videos so that it is not limited
    to the captions of the downloaded videos
'''


class Vocab:

    def __init__(self, params):

        self.params = params
        self.raw_data = read_json(params['training_data'])
        self.captions = self.raw_data['sentences']
        self.videos = self.raw_data['videos']
        self.specialWords = ['seq_start', 'seq_end', 'seq_unkown', 'seq_extra']
        self.padding_element = None

        self.vocab = dict() 
        self.vid2cap = dict() 

        self.create_vid2cap() 
        self.build_vocab()
        #self.load_embeddings()

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
        if os.path.exists(self.params['word2ix_file']) :
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

            print("vocab saved")

    def caption2seq(self, caption):
        caption = tokenize_caption(caption)[:self.params['CAPTION_LEN'] - 2]
        caption = ['seq_start'] + caption + ['seq_end']
        
        return [self.word2ix.get(word, self.word2ix['seq_unkown']) for word in caption]  

    def create_vid2cap(self):
        for video in self.videos: self.vid2cap[video['video_id']] = [video]

        for sentence in self.captions :
            if sentence['video_id'] not in self.vid2cap: continue
            self.vid2cap[sentence['video_id']].append(sentence['caption'])

    def get_caption_samples_based_on_category(self, num):
        all = [ vid  for vid in list(v.vid2cap.values()) if str(num) in str(vid[0]['category'])]
        samples = [np.random.choice(info[1:]) for info in [vid for vid in all[:30]]]
        return samples


if __name__ == "__main__":
    params = read_yaml()
    v = Vocab(params)
    print(v.vocab[-10:])
    

    
