import os
import glob
import joblib

testing_frames_cnt = []
video_num = 0
testing_video = []
dota_test = {}
video_dir = os.path.join('/data/qh/DoTA/data/metadata', 'val_split.txt')
valid_videos = []
with open(video_dir) as f:
    for row in f:
        valid_videos.append(row.strip('\n'))
for video in sorted(valid_videos):
    num_frames = len(
        glob.glob(os.path.join("/data/qh/DoTA/data/frames/", video, 'images', '*' + 'jpg')))
    testing_frames_cnt.append(num_frames)
    testing_video.append(video)
    video_num += 1
dota_test['testing_video_num'] = video_num
dota_test['testing_frames_cnt'] = testing_frames_cnt
dota_test['testing_video'] = testing_video
save_path = "/data/qh/DoTA/data/ground_truth_demo"
if not os.path.isdir(save_path):
    os.mkdir(save_path)
joblib.dump(dota_test, os.path.join(save_path, 'gt_metadata.json'))