import json 
import yaml
import time 
import random
import logging
logging.basicConfig(filename='logs/thisIsLog.log', format = '%(levelname)-8s %(asctime)s %(message)s' , level=logging.INFO)
import re
import numpy as np
import cv2 



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


def show_image(frame):
    cv2.imshow('r', frame)
    cv2.waitKey(0)  
    cv2.destroyWindow('r')



def tokenize_caption(caption):
    caption = re.sub('[^a-zA-Z]+', ' ', caption).lower()
    caption = caption.split()
    return caption


def get_embeddings(n=300):
    '''
    remember that range of embeddings generated here is [-1,1] but 
    glove embeddings range is not the same and need to be scaled
    '''
    embeds = np.random.randint(0,2, n) - np.random.rand(n)
    return embeds

