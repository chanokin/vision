import cv2
import sys
import os
import glob
import numpy as np

WHITE = [255, 255, 255]
GREEN = [0,   255,   0]
RED   = [255,   0,   0]

TOTAL_X = int(sys.argv[1])
TOTAL_Y = int(sys.argv[2])

def get_v1_videos(path=os.getcwd()):
    files = os.listdir(path)
    file_list = []
    for file in files:
        if ('v1_out_video' in file) and file.endswith(".m4v") \
            and file[0] != '.':
            # print(file)
            file_list.append(os.path.join(path, file))

    return file_list

def get_pos_from_fname(fname):
    fname_split = fname.split('_')
    r = int( fname_split[-2] )
    c = int((fname_split[-1].split('.'))[0])

    return r, c

scale = 1
vid_list = get_v1_videos()

vid_in = cv2.VideoCapture(vid_list[0])
ret, img_in = vid_in.read()
local_h, local_w, channels = img_in.shape
total_h, total_w = TOTAL_Y*local_h, TOTAL_X*local_w
frame_shape = (total_w*scale, total_h*scale)
out_shape = (total_h, total_w, channels)

scaled_out_shape = (total_w*scale, total_h*scale, channels)
img_out = np.zeros(out_shape)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
fps = 60
video_out = cv2.VideoWriter(os.path.join(os.getcwd(), "v1_full_output.m4v"),
                            fourcc, fps, frame_shape)


vid_in_list = [cv2.VideoCapture(vid_fname) for vid_fname in vid_list]

frame = 0
all_done = False
total_vids = len(vid_in_list)
while not all_done:
    sys.stdout.write("/")
    sys.stdout.flush()
    frame += 1
    vids_done = 0
    img_out[:] = 0

    for i, vid in enumerate(vid_in_list):
        ret, img_in = vid.read()
        if ret == False:
            vids_done += 1
            continue
        else:

            r, c = get_pos_from_fname(vid_list[i])
            fr_r = r * local_h
            to_r = fr_r + local_h
            fr_c = c * local_w
            to_c = fr_c + local_w

            img_out[fr_r:to_r, fr_c:to_c, :] = img_in


    video_out.write(cv2.resize(img_out, frame_shape,
                               interpolation=cv2.INTER_NEAREST).astype('uint8'))
    if vids_done == total_vids:
        break

video_out.release()
print("")