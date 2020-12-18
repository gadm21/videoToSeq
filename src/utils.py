import json 
import yaml
import time 
import random
import logging
logging.basicConfig(filename='logs/thisIsLog.log', format = '%(levelname)-8s %(asctime)s %(message)s' , level=logging.INFO)
import re
import numpy as np
#import cv2 



def log(logType, message):
    if logType == 'info' :
        logging.info( message) 
    elif logType == 'warn':
        logging.warning(message) 
    else:
        logging.debug(message) 

def get_time(): 
    return time.strftime("%H_%M_%S", time.localtime() )
    

def read_yaml(path = '7agaty.yaml'):
    
    with open(path, 'r') as stream:
        try:
            data = yaml.safe_load(stream) 
            return data
        except yaml.YAMLError as exc:
            print("error:", exc)
            return None


def read_json(path):
    with open(path) as json_file:
        data = json.load(json_file)
    
    return data 


'''
def show_image(frame):
    cv2.imshow('r', frame)
    cv2.waitKey(0)  
    cv2.destroyWindow('r')
'''


def tokenize_caption2(caption):
    stop_words = get_stop_words() 
    return list(filter(lambda x : len(x) > 2 and x not in stop_words, re.split('\W+', caption.lower())))
    
def tokenize_caption(caption):
    return list(filter(lambda x : len(x) > 1 , re.split('\W+', caption.lower())))

def get_embeddings(n=300):
    '''
    remember that range of embeddings generated here is [-1,1] but 
    glove embeddings range is not the same and need to be scaled
    '''
    embeds = np.random.randint(0,2, n) - np.random.rand(n)
    return embeds

def get_categories():
    params = read_yaml() 
    categories = [] 
    with open(params['categories_file'], 'r') as f :
        categories = f.read().split('\n') 
    return categories

def which_category(num):
    if num < 0 or num > 19 : return None 

    cats = get_categories() 
    for cat in cats : 
        if str(num) in cat :
            target = cat.split('\\')[0] 
            return target 
    return None 


def get_stop_words():
    params = read_yaml()
    words = []
    with open(params['stop_words_file'], 'r') as f : 
        words = f.read().split('\n')
    return words

if __name__=='__main__':
    cats = get_categories() 
    
    print(cats)
    for cat in cats : 
        if 'science' in cat : print('yes') 

