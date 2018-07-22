'''
A python program to to generate and save image frames from a video
'''

from __future__ import print_function, division, absolute_import
import time
import os
import numpy as np
import skvideo.io
import cv2

# ============================================================================================
# input path -- path to the video file
path_ip = "E:\\KeepVid Pro Downloaded\\Ink in Water Background (720p).mp4"

# output path -- path to save the image files
path_op = "E:\\KeepVid Pro Downloaded"

# maximum number of frames to take from the video, use "None" if you don't want to specify
# if "None", saves all frames
max_frames = None

# number of frames to skip between saving two frames of the video,
# for e.g. 0 to save all frames and 1 to save alternate frames and so on
n_skip = 0

# image extension -- "png" or "jpg" etc. The ones supported by Open cv
ext = "png"
# =============================================================================================

# create directory to save images if it does not exist
if not os.path.exists(path_op):
    os.makedirs(path_op)
    print("Directory to save images is created :", path_op)

videogen = skvideo.io.vreader(path_ip)
start_t = time.time()

n_frame = 0
n_saved = 0

for frame in videogen:
    if n_frame % (n_skip + 1) == 0:
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        path = path_op + "\\img_{0}.".format(n_saved+1) + ext
        cv2.imwrite(path, frame)
        n_saved = n_saved + 1

        if not max_frames is None:
            if n_saved == max_frames:
                break
    n_frame = n_frame + 1


print("\nTotal number of frames scanned:", n_frame + 1)
print("Total number of frames saved:", n_saved)
print("Total time taken :", time.time()-start_t)
print("Time to save per image :", (time.time()-start_t)/n_saved)
