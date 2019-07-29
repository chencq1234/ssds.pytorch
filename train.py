from __future__ import print_function

import sys
import os
import argparse
import numpy as np
if '/data/software/opencv-3.4.0/prlib/python2.7/dist-packages' in sys.path:
    sys.path.remove('/data/software/opencv-3.4.0/prlib/python2.7/dist-packages')
if '/data/software/opencv-3.3.1/prlib/python2.7/dist-packages' in sys.path:
    sys.path.remove('/data/software/opencv-3.3.1/prlib/python2.7/dist-packages')
import cv2
from datetime import datetime

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable

from lib.utils.config_parse import cfg_from_file
from lib.ssds_train import train_model

os.environ['CUDA_VISIBLE_DEVICES'] = "4,5,6,7"  # "0,1,2,3,4,5,6,7"


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Train a ssds.pytorch network')
    parser.add_argument('--cfg', dest='config_file',
            # help='optional config file', default="./experiments/cfgs/fssd_lite_mobilenetv1_train_voc-v0.5.yml", type=str)
            # help='optional config file', default="./experiments/cfgs/yolo_v3_mobilenetv2_voc.yml", type=str)
            # help='optional config file', default="./experiments/cfgs/yolo_v3_mobilenetv1_voc.yml", type=str)
            # help='optional config file', default="./experiments/cfgs/yolo_v3_mobilenetv1_voc-0.5.yml", type=str)
            # help='optional config file', default="./experiments/cfgs/ssd_lite_mobilenetv1_train_voc.yml", type=str)
            # help='optional config file', default="./experiments/cfgs/yolo_v3_mobilenetv1_voc-lite-0.5.yml", type=str)
            help='optional config file', default="./experiments/cfgs/yolo_v3_mobilenetv1_coco.yml", type=str)

    # if len(sys.argv) == 1:
    #     parser.print_help()
    #     sys.exit(1)

    args = parser.parse_args()
    return args

def train():
    args = parse_args()
    if args.config_file is not None:
        print("cfg file: %s" % args.config_file)
        cfg_from_file(args.config_file)

    train_model()

if __name__ == '__main__':

    train()
