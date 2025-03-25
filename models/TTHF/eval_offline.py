# -*- coding: utf-8 -*-
from calculate_error import *
from tadclip import *
import joblib
import argparse


parser = argparse.ArgumentParser(description='CLIP for Traffic Anomaly Detection',
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--height', type=int, default=224)  # KITTI (352, 704), DoTA (720, 1280)(480, 880), NYU (416, 544)
parser.add_argument('--width', type=int, default=224)
parser.add_argument('--normal_class', type=int, default=1)
parser.add_argument('--evaluate', type=bool, default=True)
parser.add_argument('--model_dir', type=str, default='/ssd/qh/DoTA/data/TTHF/')
parser.add_argument('--exp_name', type=str, default='TDAFF_BASE_RN50')
parser.add_argument('--other_method', type=str, default='TDAFF_BASE')
parser.add_argument('--dataset', type=str, default='dota')


def main():
    args = parser.parse_args()
    scores = joblib.load(
        open(os.path.join(args.model_dir, args.exp_name, "frame_scores_%s_%s_best.json" % (args.height, args.width)), "rb"))
    if args.dataset == 'dota':
        gt = joblib.load(
            open(os.path.join('/ssd/qh/DoTA/data/', "ground_truth_demo/gt_label.json"),
                 "rb"))
    elif args.dataset == 'dada':
        gt = joblib.load(
            open(os.path.join('/data/lrq/DADA-2000', "ground_truth_demo/gt_label.json"),
                 "rb"))
    TAD_result = compute_tad_scores(scores, gt, args, dataset=args.dataset)
    print('AUC result: ', TAD_result)


if __name__ == "__main__":
    main()



