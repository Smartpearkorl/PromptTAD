import os
import json
import joblib
import argparse

root_path = '/data/lrq/DADA-2000/DADA_dataset'  # path to DADA dataset label


def load_DADA(test_file, label_path, mode):
    valid_videos = []
    if mode == 'train':
        video_list = os.path.join(root_path, 'train_list.txt')
    elif mode == 'val':
        video_list = os.path.join(root_path, 'val_list.txt')
    else:
        video_list = os.path.join(root_path, 'test_list.txt')
    with open(test_file) as f:
        ind = json.load(f)
        for row in ind:
            class_id = row[0][0]
            video_id = row[0][1]
            video_name = class_id + '_' + video_id
            valid_videos.append(video_name)

    print("Number of testing videos: ", len(valid_videos))
    labels = {}
    video_lengths = {}
    for video in sorted(valid_videos):
        label_list = []
        anomaly = []
        video_path = video.split('_')
        try:
            with open(os.path.join(label_path, video_path[0], video_path[1], 'attach', 'anomaly.txt'),
                      'r') as f:
                for i, row in enumerate(f):
                    label = int(row.strip('\n'))
                    if label == 1:
                        anomaly.append(i)
                    label_list.append(int(row.strip('\n')))
        except:
            with open(os.path.join(label_path, video_path[0], video_path[1], 'attach',
                                   video_path[1] + '_anomaly.txt'), 'r') as f:
                for i, row in enumerate(f):
                    label = int(row.strip('\n'))
                    if label == 1:
                        anomaly.append(i)
                    label_list.append(int(row.strip('\n')))
        try:
            anomaly_start = anomaly[0]
            anomaly_end = anomaly[-1]
        except:
            print('there is no anomaly ', video)
        video_lengths[video] = len(label_list)
        labels[video] = label_list
        with open(video_list, 'a+') as f:
            f.write(video)
            f.write('\n')
    return labels


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="test", help='train, val or test data')
    args = parser.parse_args()
    if args.mode == 'train':
        video_file = '/data/lrq/DADA-2000/train_file.json'
    elif args.mode == 'val':
        video_file = '/data/lrq/DADA-2000/val_file.json'
    else:
        video_file = '/data/lrq/DADA-2000/test_file.json'
    label_path = '/data/lrq/DADA-2000/DADA_dataset'
    save_path = '/data/lrq/DADA-2000/ground_truth_demo/'
    if not os.path.isdir(save_path):
        os.mkdir(save_path)
    labels = load_DADA(video_file, label_path, args.mode)
    if args.mode == 'train':
        joblib.dump(labels, os.path.join(save_path, 'train_label.json'))
    elif args.mode == 'val':
        joblib.dump(labels, os.path.join(save_path, 'val_label.json'))
    else:
        joblib.dump(labels, os.path.join(save_path, 'gt_label.json'))
