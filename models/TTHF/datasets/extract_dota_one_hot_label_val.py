import json
import joblib
import os
import numpy as np

meta_data = json.load(
    open(os.path.join('/data/qh/DoTA/data/metadata', "metadata_val.json"), "rb"))

accident_promote_class = ['The ego vehicle collision with another vehicle',
                          'The ego vehicle collision with another pedestrian',
                          'The ego vehicle collision with another obstacle',
                          'The non-ego vehicle collision with another vehicle',
                          'The non-ego vehicle collision with another pedestrian',
                          'The non-ego vehicle collision with another obstacle',
                          'The ego vehicle out-of-control and leaving the roadway',
                          'The non-ego vehicle out-of-control and leaving the roadway',
                          'The ego vehicle has an unknown accident',
                          'The non-ego vehicle has an unknown accident',
                          'The vehicle is running normally on the road']
one_hot_label = np.eye(len(accident_promote_class))
one_hot_label_dict = {k: [] for k in meta_data.keys()}

for key in meta_data.keys():
    video_len = meta_data[key]['video_end'] - meta_data[key]['video_start'] + 1
    for idx in range(video_len):
        if idx < meta_data[key]['anomaly_start'] or idx > meta_data[key]['anomaly_end']:
            one_hot_label_dict[key].append(['{:06d}'.format(idx), 10, accident_promote_class[-1]])
        else:
            if meta_data[key]['anomaly_class'] == 'other: moving_ahead_or_waiting':
                one_hot_label_dict[key].append(['{:06d}'.format(idx), 3, accident_promote_class[3]])
            elif meta_data[key]['anomaly_class'] == 'other: turning':
                one_hot_label_dict[key].append(['{:06d}'.format(idx), 3, accident_promote_class[3]])
            elif meta_data[key]['anomaly_class'] == 'other: lateral':
                one_hot_label_dict[key].append(['{:06d}'.format(idx), 3, accident_promote_class[3]])
            elif meta_data[key]['anomaly_class'] == 'other: leave_to_left' or \
                    meta_data[key]['anomaly_class'] == 'other: leave_to_right':
                one_hot_label_dict[key].append(['{:06d}'.format(idx), 7, accident_promote_class[7]])
            elif meta_data[key]['anomaly_class'] == 'other: obstacle':
                one_hot_label_dict[key].append(['{:06d}'.format(idx), 5, accident_promote_class[5]])
            elif meta_data[key]['anomaly_class'] == 'other: start_stop_or_stationary':
                one_hot_label_dict[key].append(['{:06d}'.format(idx), 3, accident_promote_class[3]])
            elif meta_data[key]['anomaly_class'] == 'other: oncoming':
                one_hot_label_dict[key].append(['{:06d}'.format(idx), 3, accident_promote_class[3]])
            elif meta_data[key]['anomaly_class'] == 'other: pedestrian':
                one_hot_label_dict[key].append(['{:06d}'.format(idx), 4, accident_promote_class[4]])
            elif meta_data[key]['anomaly_class'] == 'other: unknown':
                one_hot_label_dict[key].append(['{:06d}'.format(idx), 9, accident_promote_class[9]])
            elif meta_data[key]['anomaly_class'] == 'ego: moving_ahead_or_waiting':
                one_hot_label_dict[key].append(['{:06d}'.format(idx), 0, accident_promote_class[0]])
            elif meta_data[key]['anomaly_class'] == 'ego: turning':
                one_hot_label_dict[key].append(['{:06d}'.format(idx), 0, accident_promote_class[0]])
            elif meta_data[key]['anomaly_class'] == 'ego: lateral':
                one_hot_label_dict[key].append(['{:06d}'.format(idx), 0, accident_promote_class[0]])
            elif meta_data[key]['anomaly_class'] == 'ego: leave_to_left' or \
                    meta_data[key]['anomaly_class'] == 'ego: leave_to_right':
                one_hot_label_dict[key].append(['{:06d}'.format(idx), 6, accident_promote_class[6]])
            elif meta_data[key]['anomaly_class'] == 'ego: obstacle':
                one_hot_label_dict[key].append(['{:06d}'.format(idx), 2, accident_promote_class[2]])
            elif meta_data[key]['anomaly_class'] == 'ego: start_stop_or_stationary':
                one_hot_label_dict[key].append(['{:06d}'.format(idx), 0, accident_promote_class[0]])
            elif meta_data[key]['anomaly_class'] == 'ego: oncoming':
                one_hot_label_dict[key].append(['{:06d}'.format(idx), 0, accident_promote_class[0]])
            elif meta_data[key]['anomaly_class'] == 'ego: pedestrian':
                one_hot_label_dict[key].append(['{:06d}'.format(idx), 1, accident_promote_class[1]])
            elif meta_data[key]['anomaly_class'] == 'ego: unknown':
                one_hot_label_dict[key].append(['{:06d}'.format(idx), 8, accident_promote_class[8]])

save_path = "/data/qh/DoTA/data/ground_truth_demo"
if not os.path.isdir(save_path):
    os.mkdir(save_path)
joblib.dump(one_hot_label_dict, os.path.join(save_path, 'val_one_hot_label.json'))

