from utils import * 
from model import VModel
from VideoDataHandler import VideoDataHandler

params = read_yaml()



def main():
    
    vmodel = VModel(params) 


if __name__ == "__main__":
    main()