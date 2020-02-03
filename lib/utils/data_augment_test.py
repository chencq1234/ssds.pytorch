"""Data augmentation functionality. Passed as callable transformations to
Dataset classes.

The data augmentation procedures were interpreted from @weiliu89's SSD paper
http://arxiv.org/abs/1512.02325

Ellis Brown, Max deGroot
"""

import cv2
import numpy as np
from data_augment import draw_bbox,_crop,_distort,_elastic,_expand,_mirror, swap_channels, RandomBrightness, RandomHue, preproc
# from data_augment import *
from lib.utils.amdegroot_augmentations import RandomRotate

if __name__ == '__main__':
    # image = cv2.imread('./experiments/2011_001100.jpg')
    # boxes = np.array([np.array([124, 150, 322, 351])])  # ymin, xmin, ymax, xmax
    rgb_means = (103.94, 116.78, 123.68)
    resize = (416, 416)
    image = cv2.imread('../../experiments/201904281045502043.jpg')
    boxes = np.array([[399, 93, 536, 110, 1], [777, 130, 1221, 195, 1]], dtype=np.float)  # ymin, xmin, ymax, xmax
    labels = boxes[:, 4]
    boxes = boxes[:, :4]
    kps = np.array([[[534, 99], [517, 108], [341, 104], [370, 95]], [[835, 193], [779, 146], [1091, 132], [1219, 159]]],
                   dtype=np.float)
    for i in range(10):
        rotate = RandomRotate(10)
        image_t, boxes, labels, kpst = rotate(image, boxes, labels, kps)
        for idx, kp in enumerate(kpst):
            for idx2, p in enumerate(kp):
                cv2.line(image_t, tuple(map(int, p)), tuple(map(int, kp[(idx2+1) % 4])), (0, 255, 0), 5)
        cv2.imshow('crop_image', image_t)
        cv2.waitKey(0)
        # cv2.imwrite("../../experiments/test_augment/rotate"+str(i)+'.jpg', image)
    # image = cv2.imread('../../experiments/person.jpg')
    # boxes = np.array([np.array([192, 99, 275, 375, 1], dtype=np.float)])  # ymin, xmin, ymax, xmax
    # labels = np.array([[1]])
    # p = 1
    # prep = preproc(resize,rgb_means, p, "../../experiments/test_augment/")
    # prep(image, boxes)
    # for i in range(20):
    #     image = cv2.imread('../../experiments/person.jpg')
    #     boxes = np.array([np.array([192, 99, 275, 375])])  # ymin, xmin, ymax, xmax
    #     labels = np.array([[1]])
    #     p = 1
    #
    #     image_show = draw_bbox(image, boxes)
    #     cv2.imshow('input_image', image_show)
    #     cv2.waitKey(0)
    #
    #     # for i in range(20):
    #     #     img = image.copy()
    #     #     box = boxes.copy()
    #     #     image_t, box, labels = _crop(image, box, labels)
    #     #     image_show = draw_bbox(image_t, box)
    #     #     cv2.imshow('crop_image', image_show)
    #     #     cv2.waitKey(0)
    #
    #     image_t, boxes, labels = _crop(image, boxes, labels)
    #     image_show = draw_bbox(image_t, boxes)
    #     cv2.imshow('crop_image', image_show)
    #     cv2.waitKey(0)
    #     # image_t, boxes, _ = RandomHue()(image_t, boxes, labels)
    #     # image_show = draw_bbox(image_t, boxes)
    #     # cv2.imshow('RandomHue', image_show)
    #     # cv2.waitKey(0)
    #     # image_t, boxes, _ = RandomBrightness()(image_t, boxes, labels)
    #     # image_show = draw_bbox(image_t, boxes)
    #     # cv2.imshow('RandomBrightness', image_show)
    #     # cv2.waitKey(0)
    #
    #     image_t = _distort(image_t, test_mode=1)
    #     image_show = draw_bbox(image_t, boxes)
    #     cv2.imshow('distort_image', image_show)
    #     cv2.waitKey(0)
    #     # image_t, boxes = _expand(image_t, boxes, (103.94, 116.78, 123.68), p)
    #     # image_show = draw_bbox(image_t, boxes)
    #     # cv2.imshow('expand_image', image_show)
    #     # cv2.waitKey(0)
    #     image_t, boxes = _mirror(image_t, boxes)
    #     image_show = draw_bbox(image_t, boxes)
    #     cv2.imshow('mirror_image', image_show)
    #     cv2.waitKey(0)
    #
    #     image_t = swap_channels(image_t, test_mode=1)
    #     image_show = draw_bbox(image_t, boxes)
    #     cv2.imshow('swap_channels', image_show)
    #     cv2.waitKey(0)
    #     image_t = _elastic(image_t, p, test_mode=1)
    #     image_show = draw_bbox(image_t, boxes)
    #     cv2.imshow('elastic_image', image_show)
    #     cv2.waitKey(0)

