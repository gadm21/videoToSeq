
from utils import *
import os
import urllib.parse
import urllib.request
from pytube import YouTube
from moviepy.editor import *


class VideoDataHandler():

    # ResNet
    SHAPE = (224, 224)

    LIMIT_FRAMES = 40
    AUDIO_FEATURE = (80, 40) #  TimeSamples, n_mfcc


    def __init__(self, params):

        self.raw_data = read_json(params['training_data'])
        
        self.vids_dir = params['vids_dir']
        self.caps_dir = params['caps_dir']
        self.categories = params['categories']


        if not os.path.exists(self.vids_dir): os.makedirs(self.vids_dir, exist_ok = True) 
        if not os.path.exists(self.caps_dir): os.makedirs(self.caps_dir, exist_ok = True) 

        #log('info', 'categories are:{}'.format(self.categories))

        self.processed = False 
        self.vid2cap = dict() 

        self.process()
    
    def getYoutubeId(self,url):
        query = urllib.parse.parse_qs(urllib.parse.urlparse(url).query)
        print(query) 
        return query['v'][0]

    def process(self):
        self.create_vid2cap()

        sample = self.vid2cap['video'+str(0)] 
        self.downloadVideo(sample[0]) 
        #print(sample[1:3])


    def downloadVideo(self, video ):
        url = video['url'] 
        sTime = video['start time'] 
        eTime = video['end time'] 
        videoPath = os.path.join(self.vids_dir, video['video_id']+'.mp4')
        videoPath2 = os.path.join(self.vids_dir, video['video_id']+'2.mp4')
        
        if os.path.exists(videoPath):
            log('info', '{} already downloaded'.format(video['video_id']))
            return

        YouTube(url).streams.first().download(videoPath)
        video = VideoFileClip(videoPath).subclip(sTime,eTime)
        video.write_videofile(videoPath2,fps=25)
        
        

    def create_vid2cap(self):

        for video in self.raw_data['videos'] :
            if video['video_id'] in self.vid2cap : 
                log('warn', '{} is repeated'.format(video['video_id']))
            self.vid2cap[video['video_id']] = [video] 

        for sentence in self.raw_data['sentences']:
            if sentence['video_id'] not in self.vid2cap:
                log('warn', '{} was not found in videos'.format(sentence['video_id']))
            self.vid2cap[sentence['video_id']].append(sentence['caption']) 
        