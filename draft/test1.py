
import numpy as np
from moviepy.editor import *


limit = 40 
video_path = r'C:\Users\gad\Desktop\repos\videoToSeq\data\videos\newthing.mp4'
new_video_path = r'C:\Users\gad\Desktop\repos\videoToSeq\data\videos\newthing.mp4'

def run_test():
    clip = VideoFileClip(video_path) 
    d = clip.duration 
    new_fps = limit // d 
    print("d:{} fps:{} new_fps:{}".format(d, clip.fps, new_fps))
    clip = clip.set_fps(new_fps) 
    print("newthing:", clip.fps) 
    clip.write_videofile(new_video_path)
    #print("done")
    
def run_test2():
    clip = VideoFileClip(video_path)
    for i in range(500):
        print('{}: {}'.format(i, clip.get_frame(i).shape))


if __name__ == '__main__':
    run_test2()