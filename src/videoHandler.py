
from utils import *
import os
import urllib.parse
import urllib.request
from pytube import YouTube
from moviepy.editor import *


'''
TODO add a function to extract audio & frames 

TODO add function get_video() which takes a video_id and does the following:
    1-  if the video is downloaded jump to step 6
    2- download the video
    3- clip the video from start_point to end_point 
    4- keep only the frames until the frame limit
    5- save the video(which is by now a number of frame = LIMIT_FRAMES)
    6- return the video, its captions, and its id
'''
class VideoHandler():

    # ResNet
    SHAPE = (224, 224)

    LIMIT_FRAMES = 40
    AUDIO_FEATURE = (80, 40) #  TimeSamples, n_mfcc


    def __init__(self, params):

        self.raw_data = read_json(params['training_data'])
        
        self.vids_dir = params['vids_dir']
        self.originals_dir = os.path.join(params['vids_dir'], 'originals')
        self.caps_dir = params['caps_dir']
        self.categories = params['categories']

        if not os.path.exists(self.vids_dir): os.makedirs(self.vids_dir, exist_ok = True) 
        if not os.path.exists(self.caps_dir): os.makedirs(self.caps_dir, exist_ok = True) 
        if not os.path.exists(self.originals_dir): os.makedirs(self.originals_dir, exist_ok = True) 

        self.downloaded = []
        self.vid2cap = dict() 
        '''
        downloaded: list()
        the first element is the video object containing url, video_id, etc..
        all following elements are captions. One caption per element 

        vid2cap: dictionary() 
        the key is the video_id. The value is a list. the content of the list is the same as self.downloaded

        The difference between downloaded and the value of vid2cap keys is that downloaded only
        has the information of already downloaded videos which can be used for training.
        '''
        self.create_vid2cap()



    def process(self):
        

        for i in range(9):

            sample = self.vid2cap['video'+str(i)] 
            f, path = self.downloadVideo(sample[0])  
            if f :
                log('debug', "{} downloaded".format(sample[0]['video_id']))
                self.downloaded.append(sample) 
            else:
                log('debug',"cannot download {} ".format(sample[0]['video_id']+'.mp4')) 


    def downloadVideo(self, videoName, url, sTime, eTime, trials =2 ):
        

        def on_downloaded(stream, fileHandle):   
            clip = VideoFileClip(fileHandle).subclip(sTime, eTime) 
            clip.write_videofile(os.path.join(self.vids_dir, videoName), fps = 25)

        try:
            yt = YouTube(url)
            yt.register_on_complete_callback(on_downloaded)
            stream = yt.streams.filter(subtype= 'mp4').first() 
            if stream is None : return False, None
            
            stream.download(self.originals_dir)
            
            return True, os.path.join(self.vids_dir, stream.title+'.mp4')
        except:
            print('failed to download:{}'.format(videoName)) 
            return False, None 

            if trials : 
                print("{} retrying...".format(videoName))
                self.downloadVideo(video, trials-1)
            else: 
                print("could not download {}".format(videoName))
                return False, None

    def get_video(self, video):

        url = video['url'] 
        sTime = video['start time'] 
        eTime = video['end time'] 
        videoName = video['video_id'] + '.mp4'
        if videoName in os.listdir(self.vids_dir) : 
            return True, os.path.join(self.vids_dir, videoName)
        
        f = self.downloadVideo(videoName, url, sTime, eTime) 
        


    def create_vid2cap(self):

        for video in self.raw_data['videos'] :
            if video['video_id'] in self.vid2cap : 
                log('warn', '{} is repeated'.format(video['video_id']))
            self.vid2cap[video['video_id']] = [video] 

        for sentence in self.raw_data['sentences']:
            if sentence['video_id'] not in self.vid2cap:
                log('warn', '{} was not found in videos'.format(sentence['video_id']))
                continue 
            self.vid2cap[sentence['video_id']].append(sentence['caption']) 



    def get_random_videos(self, n=1):
        
        #ids =[data['video_id'] for data in np.random.choice(self.raw_data['videos'], n) ]
        ids = ['video'+str(i) for i in np.arange(4)]

        video_metadata = [self.vid2cap[id][0] for id in ids]
        captions = [self.vid2cap[id][1:] for id in ids]

        #sentences = [sent['caption'] for sent in raw_data['sentences'] if sent['video_id'] == video_id]

   


if __name__ == '__main__': 
    videoHandler = VideoHandler(read_yaml()) 
    videoHandler.get_random_videos(3)