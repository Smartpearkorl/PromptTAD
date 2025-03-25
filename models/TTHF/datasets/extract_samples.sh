#!/bin/bash
# extract gt
python3 ./extract_gt_label.py  # for DoTA dataset

python3 ./extract_gt_label_dada.py  # for DADA dataset

python3 ./extract_dota_one_hot_label.py  # extract fine-grained label

python3 ./extract_dota_one_hot_label_bi.py  # extract general label

python3 ./extract_dota_one_hot_label_val.py  # extract val label

python3 ./extract_dota_metadata.py  # extract meta_data
