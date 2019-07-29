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

from lib.ssds import ObjectDetector
from lib.utils.config_parse import cfg_from_file

VOC_CLASSES = ('pbox',
        'stbox',
        'edbox')

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Demo a ssds.pytorch network')
    parser.add_argument('--cfg', dest='confg_file',help='the address of optional config file',
                        # default="./experiments/cfgs/yolo_v3_mobilenetv2_voc.yml", type=str)
                        default="./experiments/cfgs/yolo_v3_mobilenetv1_voc-0.5.yml", type=str)
    parser.add_argument('--demo', dest='demo_file',help='the address of the demo file',
                        default="/data-private/nas/pspace/inference_data/20190309090054", type=str)
    parser.add_argument('-t', '--type', dest='type', help='the type of the demo file, could be "image","image_dir", '
                        '"video", "camera" or "time" , default is "image_dir"', default='image_dir', type=str)
    parser.add_argument('-d', '--display', dest='display',help='whether display the detection result, default is False',
                        default=False, type=bool)
    parser.add_argument('-s', '--save', dest='save', help='whether write the detection result, default is False',
                        default="/data-private/nas/pspace/inference_data/inference_result/", type=str)

    # if len(sys.argv) == 1:
    #     parser.print_help()
    #     sys.exit(1)

    args = parser.parse_args()
    return args


COLORS = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
FONT = cv2.FONT_HERSHEY_SIMPLEX

def inference_img_dir(args, imgs_dir):
    if args.save:
        infer_key = imgs_dir.split("/")[-1].replace("/", "")
        model_name = args.confg_file.split("/")[-1].replace(".yml", "")
        test_img_rdir = os.path.join(args.save, infer_key)
        test_model_rdir = os.path.join(args.save, infer_key, model_name)
        test_out_img_path = os.path.join(args.save, infer_key, model_name, "det_imgs")
        pred_out_txt = os.path.join(args.save, infer_key, model_name, "pred_txt")
        model_save_path = os.path.join(args.save, infer_key, model_name, "model")
        if not os.path.exists(test_img_rdir):
            os.mkdir(test_img_rdir)
        if not os.path.exists(test_model_rdir):
            os.mkdir(test_model_rdir)
        if not os.path.exists(test_out_img_path):
            os.mkdir(test_out_img_path)
        if not os.path.exists(pred_out_txt):
            os.mkdir(pred_out_txt)
        if not os.path.exists(model_save_path):
            os.mkdir(model_save_path)

    # 1. load the configure file
    cfg_from_file(args.confg_file)
    # 2. load detector based on the configure file
    object_detector = ObjectDetector()
    for idx, img in enumerate(os.listdir(imgs_dir)):
        image_path = os.path.join(imgs_dir, img)
        # 3. load image
        image = cv2.imread(image_path)
        image256 = image[300:556]
        # 4. detect
        _labels, _scores, _coords = object_detector.predict(image256)

        # 5. draw bounding box on the image
        cxyl, stl, edl = [], [],[]
        for label, score, coords in zip(_labels, _scores, _coords):
            xmin, ymin, xmax, ymax = [int(round(i)) for i in coords.cpu().numpy()]
            cx = int(round((xmin + xmax) / 2))
            cy = int(round((ymin + ymax) / 2))
            min_d = min(xmax - xmin, ymax - ymin)
            if score >= 0.45:
                if label == 0:
                    cxyl.append([(cx, cy), [[xmin, ymin], [xmin, ymax], [xmax, ymax], [xmax, ymin]]])
                    cv2.circle(image256, (cx, cy), int(min_d / 4), (0, 0, 255), 2)
                    # cv2.rectangle(img,(xmin, ymin),(xmax, ymax),(0, 255, 255))
            if score >= 0.45:
                if label == 1:
                    stl.append((cx, cy))
                #     cv2.circle(image256, (cx, cy), 5, (255, 0, 0), 2)
                # elif label == 2:
                #     cv2.circle(image256, (cx, cy), 5, (0, 255, 0), 2)
                    # cv2.line(img, (xmin, ymin), (xmax, ymax), (155, 155, 155), 2)
                    # cv2.line(image256, (xmin + 20, ymin + 20), (xmax - 20, ymax - 20), (0, 255, 0), 2)
                # elif label == 4:
                #     # cv2.line(img, (xmin, ymin), (xmax, ymax), (0, 155, 155), 2)
                #     cv2.line(image256, (xmin + 20, ymin + 20), (xmax - 20, ymax - 20), (0, 155, 155), 2)
                # elif label == 5:
                #     cv2.circle(img256, (cx, cy), 5, (255, 255, 0))

            # cv2.rectangle(image256, (int(coords[0]), int(coords[1])), (int(coords[2]), int(coords[3])),
            #               COLORS[labels % 3], 2)
            # cv2.putText(image256, '{label}: {score:.3f}'.format(label=VOC_CLASSES[labels], score=scores),
            #             (int(coords[0]), int(coords[1])), FONT, 0.5, COLORS[labels % 3], 2)
        for idx2, cxyp in enumerate(cxyl):
            dis = []
            for idx1, stp in enumerate(stl):
                tdis = cv2.pointPolygonTest(np.array(cxyp[-1]), stp, True)
                tdis = float("inf") if tdis < 0 else tdis
                tdis2 = np.linalg.norm(np.array(stp)-np.array(cxyp[0]))
                dis.append(tdis + tdis2)
            idx = np.argmin(dis)
                # if cv2.pointPolygonTest(np.array(cxyp[-1]), stp, True):
            cv2.arrowedLine(image256, stl[idx], cxyp[0], (0, 0, 255))

        # 6. visualize result
        if args.display is True:
            cv2.imshow('result', image)
            cv2.waitKey(0)

        # 7. write result
        if args.save is not None:
            # path, _ = os.path.splitext(image_path)

            img_name = image_path.split("/")[-1]
            cv2.imwrite(os.path.join(test_out_img_path, img_name), image)
            print("save img:", img_name)

def demo(args, image_path):
    # 1. load the configure file
    cfg_from_file(args.confg_file)

    # 2. load detector based on the configure file
    object_detector = ObjectDetector()

    # 3. load image
    image = cv2.imread(image_path)
    image256 = image[300:556]
    # 4. detect
    _labels, _scores, _coords = object_detector.predict(image256)

    # 5. draw bounding box on the image
    for labels, scores, coords in zip(_labels, _scores, _coords):
        cv2.rectangle(image256, (int(coords[0]), int(coords[1])), (int(coords[2]), int(coords[3])), COLORS[labels % 3], 2)
        cv2.putText(image256, '{label}: {score:.3f}'.format(label=VOC_CLASSES[labels], score=scores), (int(coords[0]), int(coords[1])), FONT, 0.5, COLORS[labels % 3], 2)
    
    # 6. visualize result
    if args.display is True:
        cv2.imshow('result', image)
        cv2.waitKey(0)

    # 7. write result
    if args.save is True:
        # path, _ = os.path.splitext(image_path)

        img_name = image_path.split("/")[-1]
        cv2.imwrite(os.path.join(args.test_out_img_path, img_name), image)
    

def demo_live(args, video_path):
    # 1. load the configure file
    cfg_from_file(args.confg_file)

    # 2. load detector based on the configure file
    object_detector = ObjectDetector()

    # 3. load video
    video = cv2.VideoCapture(video_path)

    index = -1
    while(video.isOpened()):
        index = index + 1
        sys.stdout.write('Process image: {} \r'.format(index))
        sys.stdout.flush()

        # 4. read image
        flag, image = video.read()
        if flag == False:
            print("Can not read image in Frame : {}".format(index))
            break

        # 5. detect
        _labels, _scores, _coords = object_detector.predict(image)

        # 6. draw bounding box on the image
        for labels, scores, coords in zip(_labels, _scores, _coords):
            cv2.rectangle(image, (int(coords[0]), int(coords[1])), (int(coords[2]), int(coords[3])), COLORS[labels % 3], 2)
            cv2.putText(image, '{label}: {score:.3f}'.format(label=VOC_CLASSES[labels], score=scores), (int(coords[0]), int(coords[1])), FONT, 0.5, COLORS[labels % 3], 2)
    
        # 7. visualize result
        if args.display is True:
            cv2.imshow('result', image)
            cv2.waitKey(33)

        # 8. write result
        if args.save is True:
            path, _ = os.path.splitext(video_path)
            path = path + '_result'
            if not os.path.exists(path):
                os.mkdir(path)
            cv2.imwrite(path + '/{}.jpg'.format(index), image)        


def time_benchmark(args, image_path):
    # 1. load the configure file
    cfg_from_file(args.confg_file)

    # 2. load detector based on the configure file
    object_detector = ObjectDetector()

    # 3. load image
    image = cv2.imread(image_path)

    # 4. time test
    warmup = 20
    time_iter = 100
    print('Warmup the detector...')
    _t = list()
    for i in range(warmup+time_iter):
        _, _, _, (total_time, preprocess_time, net_forward_time, detect_time, output_time) \
            = object_detector.predict(image, check_time=True)
        if i > warmup:
            _t.append([total_time, preprocess_time, net_forward_time, detect_time, output_time])
            if i % 20 == 0: 
                print('In {}\{}, total time: {} \n preprocess: {} \n net_forward: {} \n detect: {} \n output: {}'.format(
                    i-warmup, time_iter, total_time, preprocess_time, net_forward_time, detect_time, output_time
                ))
    total_time, preprocess_time, net_forward_time, detect_time, output_time = np.sum(_t, axis=0)/time_iter * 1000 # 1000ms to 1s
    print('In average, total time: {}ms \n preprocess: {}ms \n net_forward: {}ms \n detect: {}ms \n output: {}ms'.format(
        total_time, preprocess_time, net_forward_time, detect_time, output_time
    ))
    with open('./time_benchmark.csv', 'a') as f:
        f.write("{:s},{:.2f}ms,{:.2f}ms,{:.2f}ms,{:.2f}ms,{:.2f}ms\n".format(args.confg_file, total_time, preprocess_time, net_forward_time, detect_time, output_time))


    
if __name__ == '__main__':
    args = parse_args()
    if args.type == 'image':
        demo(args, args.demo_file)
    elif args.type == 'image_dir':
        inference_img_dir(args, args.demo_file)
    elif args.type == 'video':
        demo_live(args, args.demo_file)
    elif args.type == 'camera':
        demo_live(args, int(args.demo_file))
    elif args.type == 'time':
        time_benchmark(args, args.demo_file)
    else:
        AssertionError('type is not correct')
