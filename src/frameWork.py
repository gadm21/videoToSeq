

from utils import * 
from videoHandler import * 
from vocab import * 
from vmodel import *

class FrameWork():

    def __init__(self, params):
        
        self.params = params 
        v_handler = videoHandler(params) 
        vocab = Vocab(params)
        model = VModel(parms) 


    