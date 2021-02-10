
from utils import *
import os
import urllib.parse
import urllib.request
from pytube import YouTube
from moviepy.editor import *
import math
import cv2
import time 

'''

    def create_vid2cap(self):

        for video in self.raw_data['videos']:
            if video['video_id'] in self.vid2cap:
                log('warn', '{} is repeated'.format(video['video_id']))
            self.vid2cap[video['video_id']] = [video]

        for sentence in self.raw_data['sentences']:
            if sentence['video_id'] not in self.vid2cap:
                log('warn', '{} was not found in videos'.format(
                    sentence['video_id']))
                continue
            self.vid2cap[sentence['video_id']].append(sentence['caption'])


'''


class VideoHandler():
    

    def __init__(self, params):

        self.params = params
        self.frame_size = (self.params['FRAME_SIZE'], self.params['FRAME_SIZE'])
        self.frames_num = self.params['FRAMES_LIMIT']
        self.raw_data = read_json(params['training_data'])
        

        self.vids_dir = params['vids_dir']
        self.caps_dir = params['caps_dir']

        os.makedirs(self.vids_dir, exist_ok=True)
        os.makedirs(self.caps_dir, exist_ok=True)

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
        #self.create_vid2cap()

    def downloadVideo(self, videoPath, url, sTime, eTime, trials=1):

        waitingTime = np.random.randint(3, 10)
        time.sleep(waitingTime)
        def on_downloaded(stream, fileHandle):
            original_clip = VideoFileClip(fileHandle)
            clip = original_clip.subclip(sTime, eTime)
            
            new_fps = math.ceil(self.params['FRAMES_LIMIT'] / clip.duration)
            clip.write_videofile(videoPath, fps=new_fps, logger=None)

            original_clip.close()
            clip.close()
            os.remove(fileHandle)

        try:
            yt = YouTube(url)
            yt.register_on_complete_callback(on_downloaded)
            stream = yt.streams.filter(subtype='mp4').first()
            if stream is None:
                return False, False 
            stream.download(self.originals_dir)
            return True, False 

        except Exception as e:
            if 'http' in str(e).lower() : return False, True
            else: return False, False

            #if trials: self.downloadVideo(videoPath, url, sTime, eTime, trials-1)
            #else: return False

    def get_video(self, video, download=False):

        url = video['url']
        sTime = video['start time']
        eTime = video['end time']
        videoName = video['video_id'] + '.mp4'
        #print("getting video:{}".format(videoName))

        videoPath = os.path.join(self.vids_dir, videoName)
        if os.path.exists(videoPath): return VideoFileClip(videoPath) 
        elif not download: return True 
        
        #print("downloading video:{}".format(videoName))
        exist, http_flag = self.downloadVideo(videoPath, url, sTime, eTime) 
        if exist : return VideoFileClip(videoPath) 
        
        if http_flag : return False
        else: return True 

    def get_video_by_id(self, video_id):
        video_name = video_id + '.mp4' 
        if video_name not in os.listdir(self.vids_dir): return None 
        video_path = os.path.join(self.vids_dir, video_name) 
        video = VideoFileClip(video_path) 
        new_fps = math.ceil(self.frames_num / video.duration)
        video = video.set_fps(new_fps)
        
        return np.array([cv2.resize(frame, self.frame_size) for frame in video.iter_frames()][:self.frames_num])





    def get_random_videos(self, n=1):
        
        ids = [id[:-4] for id in os.listdir(self.vids_dir)]

        return [self.get_video_by_id(id) for id in ids] 

        videos_metadata = [self.vid2cap[id][0] for id in ids]
        captions = [self.vid2cap[id][1:] for id in ids]


        video_clips = []
        captions = []

        for video_metadata in videos_metadata:
            video = self.get_video(video_metadata)
            all_captions = self.vid2cap[video_metadata['video_id']][1:]

            if isinstance(video, bool): 
                if video :
                    newkey = np.random.choice(list(self.vid2cap.keys()))
                    videos_metadata.append( self.vid2cap[newkey][0])
                else :
                    raise Exception('Sorry, I cannot download more videos because YouTube thinks I am a robot, I AM ')
            else :
                video_clips.append(video)
                captions.append(np.random.choice(all_captions))
                

        videos = [np.array([cv2.resize(frame, self.frame_size) for frame in video.iter_frames()][:20]) for video in video_clips]


        return videos, captions


















if __name__ == '__main__':

    video_id = 'video8331'
    frames_dir = r'C:\Users\gad\Desktop\repos\videoToSeq\frames'
    videoHandler = VideoHandler(read_yaml())
    frames = list(videoHandler.get_video_by_id(video_id))
    
    for i, frame in enumerate(frames):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        cv2.imwrite(os.path.join(frames_dir, str(i)+'.jpg'), frame) 

    print("done")
    