import json
import joblib
import os
import numpy as np

meta_data = json.load(
    open(os.path.join('/data/qh/DoTA/data/metadata', "metadata_train.json"), "rb"))

accident_promote_class = ['A traffic anomaly occurred in the scene',
                          'The traffic in this scenario is normal']
one_hot_label = np.eye(len(accident_promote_class))
one_hot_label_dict = {k: [] for k in meta_data.keys()}

for key in meta_data.keys():
    video_len = meta_data[key]['video_end'] - meta_data[key]['video_start'] + 1
    for idx in range(video_len):
        if idx < meta_data[key]['anomaly_start'] or idx > meta_data[key]['anomaly_end']:
            one_hot_label_dict[key].append(['{:06d}'.format(idx), 1, accident_promote_class[1]])
        else:
            one_hot_label_dict[key].append(['{:06d}'.format(idx), 0, accident_promote_class[0]])

save_path = "/data/qh/DoTA/data/ground_truth_demo"
if not os.path.isdir(save_path):
    os.mkdir(save_path)
joblib.dump(one_hot_label_dict, os.path.join(save_path, 'train_one_hot_label_bi.json'))

