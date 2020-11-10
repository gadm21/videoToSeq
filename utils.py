import json 
import yaml
import time 
import random
import logging
logging.basicConfig(filename='logs/thisIsLog.log', format = '%(levelname)-8s %(asctime)s %(message)s' , level=logging.INFO)


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



def get_sample(raw_data):
    
    sample = random.choice(raw_data['videos']) 
    video_id = sample['video_id']

    sentences = [sent['caption'] for sent in raw_data['sentences'] if sent['video_id'] == video_id]

    return sample, sentences 