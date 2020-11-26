from utils import * 
from vmodel import VModel
from videoHandler import videoHandler

params = read_yaml()



def main():
    
    vmodel = VModel(params) 


if __name__ == "__main__":
    main()