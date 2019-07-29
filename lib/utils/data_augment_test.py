"""Data augmentation functionality. Passed as callable transformations to
Dataset classes.

The data augmentation procedures were interpreted from @weiliu89's SSD paper
http://arxiv.org/abs/1512.02325

Ellis Brown, Max deGroot
"""

import cv2
import numpy as np
from data_augment import draw_bbox,_crop,_distort,_elastic,_expand,_mirror, swap_channels, RandomBrightness, RandomHue
# from data_augment import *

if __name__ == '__main__':
    # image = cv2.imread('./experiments/2011_001100.jpg')
    # boxes = np.array([np.array([124, 150, 322, 351])])  # ymin, xmin, ymax, xmax
    for i in range(20):
        image = cv2.imread('../../experiments/person.jpg')
        boxes = np.array([np.array([192, 99, 275, 375])])  # ymin, xmin, ymax, xmax

        labels = np.array([[1]])
        p = 1

        image_show = draw_bbox(image, boxes)
        cv2.imshow('input_image', image_show)
        cv2.waitKey(0)

        # for i in range(20):
        #     img = image.copy()
        #     box = boxes.copy()
        #     image_t, box, labels = _crop(image, box, labels)
        #     image_show = draw_bbox(image_t, box)
        #     cv2.imshow('crop_image', image_show)
        #     cv2.waitKey(0)

        image_t, boxes, labels = _crop(image, boxes, labels)
        image_show = draw_bbox(image_t, boxes)
        cv2.imshow('crop_image', image_show)
        cv2.waitKey(0)
        # image_t, boxes, _ = RandomHue()(image_t, boxes, labels)
        # image_show = draw_bbox(image_t, boxes)
        # cv2.imshow('RandomHue', image_show)
        # cv2.waitKey(0)
        # image_t, boxes, _ = RandomBrightness()(image_t, boxes, labels)
        # image_show = draw_bbox(image_t, boxes)
        # cv2.imshow('RandomBrightness', image_show)
        # cv2.waitKey(0)

        image_t = _distort(image_t)
        image_show = draw_bbox(image_t, boxes)
        cv2.imshow('distort_image', image_show)
        cv2.waitKey(0)
        # image_t, boxes = _expand(image_t, boxes, (103.94, 116.78, 123.68), p)
        # image_show = draw_bbox(image_t, boxes)
        # cv2.imshow('expand_image', image_show)
        # cv2.waitKey(0)
        image_t, boxes = _mirror(image_t, boxes)
        image_show = draw_bbox(image_t, boxes)
        cv2.imshow('mirror_image', image_show)
        cv2.waitKey(0)

        image_t = swap_channels(image_t)
        image_show = draw_bbox(image_t, boxes)
        cv2.imshow('swap_channels', image_show)
        cv2.waitKey(0)
        image_t = _elastic(image_t, p)
        image_show = draw_bbox(image_t, boxes)
        cv2.imshow('elastic_image', image_show)
        cv2.waitKey(0)

