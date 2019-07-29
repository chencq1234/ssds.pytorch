from __future__ import print_function

import sys
import os
import argparse
import numpy as np

from lib.dataset.voc import VOC_CLASSES

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
from lib.ssds_train import test_model

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Train a ssds.pytorch network')
    parser.add_argument('--cfg', dest='config_file',
            help='optional config file', default="./experiments/cfgs/yolo_v3_mobilenetv1_voc-0.5.yml", type=str)
            # help='optional config file', default="./experiments/cfgs/yolo_v3_mobilenetv2_voc.yml", type=str)

    # if len(sys.argv) == 1:
    #     parser.print_help()
    #     sys.exit(1)

    args = parser.parse_args()
    return args

def test():
    args = parse_args()
    if args.config_file is not None:
        cfg_from_file(args.config_file)
    test_model()


def inference(model, img_path):
    # for i in iter(range((num_images))):
    img = dataset.pull_image(i)
    img_key = dataset.ids[i][-1]
    scale = [img.shape[1], img.shape[0], img.shape[1], img.shape[0]]
    if use_gpu:
        images = Variable(dataset.preproc(img)[0].unsqueeze(0).cuda(), volatile=True)
    else:
        images = Variable(dataset.preproc(img)[0].unsqueeze(0), volatile=True)

    _t.tic()
    # forward
    out = model(images, phase='eval')

    # detect
    detections = detector.forward(out)

    time = _t.toc()

    # TODO: make it smart:
    pr_arr = []
    for j in range(1, num_classes):
        cls_dets = list()
        cls_name = VOC_CLASSES[j]
        for det in detections[0][j]:
            if det[0] > 0:
                d = det.cpu().numpy()
                score, box = d[0], d[1:]
                box *= scale
                box = np.append(box, score)
                cls_dets.append(box)
                if need_pr:
                    det_res4pr = [cls_name] + [str(i) for i in d]
                    pr_arr.append(' '.join(det_res4pr))

        if len(cls_dets) == 0:
            cls_dets = empty_array
        all_boxes[j][i] = np.array(cls_dets)
    if need_pr:
        pr_res_str = '\n'.join(pr_arr)
        with open(os.path.join(pr_path, img_key + ".txt"), "w") as f:
            f.write(pr_res_str)
    # log per iter
    log = '\r==>Test: || {iters:d}/{epoch_size:d} in {time:.3f}s [{prograss}]\r'.format(
        prograss='#' * int(round(10 * i / num_images)) + '-' * int(round(10 * (1 - i / num_images))), iters=i,
        epoch_size=num_images,
        time=time)
    # print(log)
    sys.stdout.write(log)
    sys.stdout.flush()


if __name__ == '__main__':
    test()
