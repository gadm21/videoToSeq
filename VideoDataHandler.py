
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
        self.originals_dir = os.path.join(params['vids_dir'], 'originals')
        self.caps_dir = params['caps_dir']
        self.categories = params['categories']


        if not os.path.exists(self.vids_dir): os.makedirs(self.vids_dir, exist_ok = True) 
        if not os.path.exists(self.caps_dir): os.makedirs(self.caps_dir, exist_ok = True) 
        if not os.path.exists(self.originals_dir): os.makedirs(self.originals_dir, exist_ok = True) 

        self.vid2cap = dict() 

        #self.process()

    def getYoutubeId(self,url):
        query = urllib.parse.parse_qs(urllib.parse.urlparse(url).query)
        return query['v'][0]

    def process(self):
        self.create_vid2cap()

        for i in range(10):

            sample = self.vid2cap['video'+str(i)] 
            f, path = self.downloadVideo(sample[0]) 
            if not f :
                print("{} is already downloaded".format(sample[0]['video_id']+'.mp4'))
            else:
                print("{} downloaded".format(sample[0]['video_id']+'.mp4')) 


    def downloadVideo(self, video, trials =2 ):
        url = video['url'] 
        sTime = video['start time'] 
        eTime = video['end time'] 
        videoName = video['video_id'] + '.mp4'
        if videoName in os.listdir(self.vids_dir) : return False, None

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
            if trials : 
                print("{} retrying...".format(videoName))
                self.downloadVideo(video, trials-1)
            else: 
                print("could not download {}".format(videoName))
                return False, None



    def create_vid2cap(self):

        for video in self.raw_data['videos'] :
            if video['video_id'] in self.vid2cap : 
                log('warn', '{} is repeated'.format(video['video_id']))
            self.vid2cap[video['video_id']] = [video] 

        for sentence in self.raw_data['sentences']:
            if sentence['video_id'] not in self.vid2cap:
                log('warn', '{} was not found in videos'.format(sentence['video_id']))
            self.vid2cap[sentence['video_id']].append(sentence['caption']) 
        