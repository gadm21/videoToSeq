from utils import * 
#from model import * 
from VideoDataHandler import VideoDataHandler

params = read_yaml()



def main():
    

    V = VideoDataHandler(params) 
    sample = list(V.raw_data['videos'])[600]
    video_id = sample['video_id']
    sentences = [sent['caption'] for sent in V.raw_data['sentences'] if sent['video_id']==video_id]
    [print(s) for s in sentences]

if __name__ == "__main__":
    main()