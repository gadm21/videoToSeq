
import numpy as np
from moviepy.editor import *
import math
import os 
import sys
sys.path.append(r'C:\Users\gad\Desktop\repos\videoToSeq\src')
import nltk
from textblob import TextBlob, Word 
from collections import Counter
from utils import *

limit = 40 
video_path = r'C:\Users\gad\Desktop\repos\videoToSeq\data\videos\newthing.mp4'
new_video_path = r'C:\Users\gad\Desktop\repos\videoToSeq\data\videos\newthing.mp4'



def run_test(videos, captions):
    
    relevant_words = ['class', 'student', 'teacher', 'class'] 
    irrelevant_words = ['truck', 'road']
    relevant = findWholeWords(relevant_words) 
    irrelevant = findWholeWords(irrelevant_words)
    captions = [(caption['caption'], caption['video_id']) for caption in captions if searchForWord(relevant, caption['caption']) and not searchForWord(irrelevant, caption['caption'])]
    #print("captions related to classroom:", len(captions))

    all_nouns_verbs = []
    new_captions = []
    videos = []
    words = []
    for caption, video_id in captions:
        tokens = tokenize_caption(caption) 
        nouns_verbs = [(Word(word).lemmatize(myPosTagger(pos)), myPosTagger(pos)) for (word, pos) in nltk.pos_tag(tokens) if pos[:2]=='NN' or pos[:2]=='VB']
        if len(nouns_verbs) != 3 or not SOV(nouns_verbs): continue 
        if video_id in videos :continue
        videos.append(video_id)
        all_nouns_verbs.append(nouns_verbs)
        new_captions.append(caption) 
        words.extend(tokens)
        
    #print("captions:", len(all_nouns_verbs))
    #[print(noun_verb) for noun_verb in all_nouns_verbs[:3]]
    #print()
    #[print(caption) for caption in new_captions[:3]]

    freqs = Counter(words)
    freqs = sorted(freqs.items(), key = lambda item: item[1])
    #print(freqs)
    print(len(words))
    print(words)


if __name__ == '__main__':
    
    params = read_yaml()
    raw_data = read_json(params['training_data'])
    captions = raw_data['sentences']
    videos = raw_data['videos']
    run_test(videos, captions)
    