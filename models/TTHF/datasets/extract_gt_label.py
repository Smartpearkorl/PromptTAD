import os
import numpy as np
import json
import joblib
import json
root_path = '/data/qh/DoTA/data/metadata/'


def load_DoTA(label_file):
    '''
    DoTA labels are saved as pkl
    '''
    valid_videos = []
    video_dir = os.path.join(root_path, 'val_split.txt')  # path to val split
    video_list = os.path.join(root_path, 'val_list.txt')
    with open(video_dir) as f:
        for row in f:
            valid_videos.append(row.strip('\n'))
    full_labels = json.load(open(label_file, 'rb'))
    print("Number of testing videos: ", len(full_labels.keys()))
    labels = {}
    video_lengths = {}
    for video_name in sorted(valid_videos):
        value = full_labels[video_name]
        video_lengths[video_name] = int(value['video_end']) - int(value['video_start']) + 1
        start = value['anomaly_start']
        end = value['anomaly_end']
        label = np.zeros(video_lengths[video_name])
        label[start: end + 1] = 1
        labels[video_name] = label
        with open(video_list, 'a+') as f:
            f.write(video_name)
            f.write('\n')
    return labels


if __name__=='__main__':
    label_file = "/data/qh/DoTA/data/metadata/metadata_val.json"
    save_path = "/data/qh/DoTA/data/ground_truth_demo"
    if not os.path.isdir(save_path):
        os.mkdir(save_path)
    labels = load_DoTA(label_file)
    joblib.dump(labels, os.path.join(save_path, 'gt_label.json'))
