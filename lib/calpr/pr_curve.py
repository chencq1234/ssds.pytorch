# encoding=utf-8
###########################################################################################
#                                                                                         #
# This sample shows how to evaluate object pr_detections applying the following metrics:     #
#  * Precision x Recall curve       ---->       used by VOC PASCAL 2012)                  #
#  * Average Precision (AP)         ---->       used by VOC PASCAL 2012)                  #
#                                                                                         #
# Developed by: Rafael Padilla (rafael.padilla@smt.ufrj.br)                               #
#        SMT - Signal Multimedia and Telecommunications Lab                               #
#        COPPE - Universidade Federal do Rio de Janeiro                                   #
#        Last modification: Oct 9th 2018                                                 #
###########################################################################################

import argparse
import glob
import os
import shutil
# from argparse import RawTextHelpFormatter
import sys

from lib.BoundingBox import BoundingBox
from lib.BoundingBoxes import BoundingBoxes
from lib.Evaluator import *
from lib.utils import BBFormat
import shelve

def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)


# currentPath = os.path.dirname(os.path.realpath(__file__))

# Add lib to PYTHONPATH
# libPath = os.path.join('lib')
# add_path(libPath)

# Validate formats
def ValidateFormats(argFormat, argName, errors):
    if argFormat == 'xywh':
        return BBFormat.XYWH
    elif argFormat == 'xyrb':
        return BBFormat.XYX2Y2
    elif argFormat is None:
        return BBFormat.XYWH  # default when nothing is passed
    else:
        errors.append(
            'argument %s: invalid value. It must be either \'xywh\' or \'xyrb\'' % argName)


# Validate mandatory args
def ValidateMandatoryArgs(arg, argName, errors):
    if arg is None:
        errors.append('argument %s: required argument' % argName)
    else:
        return True


def ValidateImageSize(arg, argName, argInformed, errors):
    errorMsg = 'argument %s: required argument if %s is relative' % (argName, argInformed)
    ret = None
    if arg is None:
        errors.append(errorMsg)
    else:
        arg = arg.replace('(', '').replace(')', '')
        args = arg.split(',')
        if len(args) != 2:
            errors.append(
                '%s. It must be in the format \'width,height\' (e.g. \'600,400\')' % errorMsg)
        else:
            if not args[0].isdigit() or not args[1].isdigit():
                errors.append(
                    '%s. It must be in INdiaTEGER the format \'width,height\' (e.g. \'600,400\')' %
                    errorMsg)
            else:
                ret = (int(args[0]), int(args[1]))
    return ret


#
def ValidateCoordinatesTypes(arg, argName, errors):
    """
    Validate coordinate types 确定 xywh 为相对于图片尺寸，还是绝对的像素值, 一般xyxy为绝对abs，wywh为相对rel
    :param arg:
    :param argName:
    :param errors:
    :return:
    """
    if arg == 'abs':
        return CoordinatesType.Absolute
    elif arg == 'rel':
        return CoordinatesType.Relative
    elif arg is None:
        return CoordinatesType.Absolute  # default when nothing is passed
    errors.append('argument %s: invalid value. It must be either \'rel\' or \'abs\'' % argName)


def ValidatePaths(arg, nameArg, errors):
    if arg is None:
        errors.append('argument %s: invalid directory' % nameArg)
    elif os.path.isdir(arg) is False:
        errors.append('argument %s: directory does not exist \'%s\'' % (nameArg, arg))
    # elif os.path.isdir(os.path.join(currentPath, arg)) is True:
    #     arg = os.path.join(currentPath, arg)
    else:
        arg = os.path.join(arg)
    return arg


def getBoundingBoxes(directory,
                     isGT,
                     bbFormat,
                     coordType,
                     allBoundingBoxes=None,
                     allClasses=None,
                     imgSize=(0, 0)):
    """Read txt files containing bounding boxes (ground truth and pr_detections)."""
    if allBoundingBoxes is None:
        allBoundingBoxes = BoundingBoxes()
    if allClasses is None:
        allClasses = []
    # Read ground truths
    os.chdir(directory)
    files = glob.glob("*.txt")
    files.sort()
    # Read GT pr_detections from txt file
    # Each line of the files in the pr_groundtruths folder represents a ground truth bounding box
    # (bounding boxes that a detector should detect)
    # Each value of each line is  "class_id, x, y, width, height" respectively
    # Class_id represents the class of the bounding box
    # x, y represents the most top-left coordinates of the bounding box
    # x2, y2 represents the most bottom-right coordinates of the bounding box
    for f in files:
        nameOfImage = f.replace(".txt", "")
        fh1 = open(f, "r")
        # fh1 = fh1.readlines()
        for line in fh1:
            line = line.replace("\n", "")
            if line.replace(' ', '') == '':
                continue
            splitLine = line.split(" ")
            if isGT:
                # idClass = int(splitLine[0]) #class
                idClass = (splitLine[0])  # class
                x = float(splitLine[1])
                y = float(splitLine[2])
                w = float(splitLine[3])
                h = float(splitLine[4])
                bb = BoundingBox(
                    nameOfImage,
                    idClass,
                    x,
                    y,
                    w,
                    h,
                    coordType,
                    imgSize,
                    BBType.GroundTruth,
                    format=bbFormat)
            else:
                # idClass = int(splitLine[0]) #class
                idClass = (splitLine[0])  # class
                confidence = float(splitLine[1])
                x = float(splitLine[2])
                y = float(splitLine[3])
                w = float(splitLine[4])
                h = float(splitLine[5])
                bb = BoundingBox(
                    nameOfImage,
                    idClass,
                    x,
                    y,
                    w,
                    h,
                    coordType,
                    imgSize,
                    BBType.Detected,
                    confidence,
                    format=bbFormat)
            allBoundingBoxes.addBoundingBox(bb)
            if idClass not in allClasses:
                allClasses.append(idClass)
        fh1.close()
    return allBoundingBoxes, allClasses

def process_pr_curve(args):
# iouThreshold default=0.5
    iouThreshold = args.iouThreshold

    # Arguments validation
    errors = []
    # Validate formats
    # XYWH = 1  XYX2Y2 = 2 default is 1: xywh
    gtFormat = ValidateFormats(args.gtFormat, '-gtformat', errors)
    detFormat = ValidateFormats(args.detFormat, '-detformat', errors)
    # Groundtruth folder
    if ValidateMandatoryArgs(args.gtFolder, '-gt/--gtfolder', errors):
        # args.gtFolder应为相对路径
        gtFolder = ValidatePaths(args.gtFolder, '-gt/--gtfolder', errors)
    else:
        # errors.pop() 默认为groundtruths文件夹
        gtFolder = os.path.join('pr_groundtruths')
        if os.path.isdir(gtFolder) is False:
            errors.append('folder %s not found' % gtFolder)

    # Coordinates types 确定坐标为相对还是绝对，一般xyxy为绝对abs，wywh为相对rel
    gtCoordType = ValidateCoordinatesTypes(args.gtCoordinates, '-gtCoordinates', errors)
    detCoordType = ValidateCoordinatesTypes(args.detCoordinates, '-detCoordinates', errors)
    imgSize = (0, 0)

    # 相对坐标需要获取图像尺寸mage size. Required if -gtcoords or -detcoords are \'rel\
    if gtCoordType == CoordinatesType.Relative:  # Image size is required
        imgSize = ValidateImageSize(args.imgSize, '-imgsize', '-gtCoordinates', errors)
    if detCoordType == CoordinatesType.Relative:  # Image size is required
        imgSize = ValidateImageSize(args.imgSize, '-imgsize', '-detCoordinates', errors)

    # Detection folder detFolder路径为相对路径
    if ValidateMandatoryArgs(args.detFolder, '-det/--detfolder', errors):
        detFolder = ValidatePaths(args.detFolder, '-det/--detfolder', errors)
    else:
        # errors.pop()
        detFolder = os.path.join('pr_detections')
        if os.path.isdir(detFolder) is False:
            errors.append('folder %s not found' % detFolder)
    if args.savePath is not None:
        os.makedirs(args.savePath, exist_ok=True)
        savePath = ValidatePaths(args.savePath, '-sp/--savepath', errors)
    else:
        savePath = args.detFolder[:-1] if args.detFolder.endswith('/') else args.detFolder
        savePath = "".join([savePath, "prApRes"])

        # savePath[-1] = "".join([savePath[-1], "prApRes"])
        # savePath = os.path.join(savePath)
        os.makedirs(savePath, exist_ok=True)
    # Validate savePath
    # If error, show error messages
    if len(errors) is not 0:
        print("""usage: Object Detection Metrics [-h] [-v] [-gt] [-det] [-t] [-gtformat]
                                    [-detformat] [-save]""")
        print('Object Detection Metrics: error(s): ')
        [print(e) for e in errors]
        sys.exit()

    # Create directory to save results 保存结果
    shutil.rmtree(savePath, ignore_errors=True)  # Clear folder
    os.makedirs(savePath)
    # Show plot during execution执行中显示结果
    showPlot = args.showPlot

    # print('iouThreshold= %f' % iouThreshold)
    # print('savePath = %s' % savePath)
    # print('gtFormat = %s' % gtFormat)
    # print('detFormat = %s' % detFormat)
    # print('gtFolder = %s' % gtFolder)
    # print('detFolder = %s' % detFolder)
    # print('gtCoordType = %s' % gtCoordType)
    # print('detCoordType = %s' % detCoordType)
    # print('showPlot %s' % showPlot)

    # Get groundtruth boxes
    allBoundingBoxes, allClasses = getBoundingBoxes(
        gtFolder, True, gtFormat, gtCoordType, imgSize=imgSize)
    # Get detected boxes
    allBoundingBoxes, allClasses = getBoundingBoxes(
        detFolder, False, detFormat, detCoordType, allBoundingBoxes, allClasses, imgSize=imgSize)
    allClasses.sort()

    evaluator = Evaluator()
    acc_AP = 0
    validClasses = 0

    # Plot Precision x Recall curve
    # method_dict = {1: "EveryPointInterpolation", 2: "ElevenPointInterpolation"}
    method = args.method
    # print(f"iouThreshold is: {iouThreshold}  |  Interpolation method: {method.name}")
    print("iouThreshold is: %s  |  Interpolation method: %s" % (iouThreshold, method.name))
    detections = evaluator.PlotPrecisionRecallCurve(
        allBoundingBoxes,  # Object containing all bounding boxes (ground truths and pr_detections)
        IOUThreshold=iouThreshold,  # IOU threshold
        # method=MethodAveragePrecision.ElevenPointInterpolation,
        method=method,
        showAP=True,  # Show Average Precision in the title of the plot
        showInterpolatedPrecision=False,  # Don't plot the interpolated precision curve
        savePath=savePath,
        showGraphic=showPlot)

    f = open(os.path.join(savePath, 'results.txt'), 'w')
    # f.write('Object Detection Metrics\n')
    # f.write('https://github.com/rafaelpadilla/Object-Detection-Metrics\n\n\n')
    f.write('Average Precision (AP), Precision and Recall per class:')

    # each detection is a class
    res_dict = {}
    for metricsPerClass in detections:

        # Get metric values per each class
        cl = metricsPerClass['class']
        res_dict[cl] = {}
        ap = metricsPerClass['AP']
        precision = metricsPerClass['precision']
        recall = metricsPerClass['recall']
        totalPositives = metricsPerClass['total positives']
        total_TP = metricsPerClass['total TP']
        total_FP = metricsPerClass['total FP']

        if totalPositives > 0:
            validClasses = validClasses + 1
            acc_AP = acc_AP + ap
            prec = [p for p in precision]
            rec = [r for r in recall]
            # prec = ['%.2f' % p for p in precision]
            # rec = ['%.2f' % r for r in recall]
            ap_str = "{0:.2f}%".format(ap * 100)
            # ap_str = "{0:.4f}%".format(ap * 100)
            print('AP: %s (%s)' % (ap_str, cl))
            f.write('\n\nClass: %s' % cl)
            f.write('\nAP: %s' % ap_str)
            f.write('\nPrecision: %s' % prec)
            f.write('\nRecall: %s' % rec)
            res_dict[cl]['ap'] = ap_str
            res_dict[cl]['p'] = prec
            res_dict[cl]['r'] = rec

    mAP = acc_AP / validClasses
    mAP_str = "{0:.2f}%".format(mAP * 100)
    print('mAP: %s' % mAP_str)
    f.write('\n\n\nmAP: %s' % mAP_str)
    f.close()
    res_dict['map'] = mAP_str
    if args.save_pr_shelve:
        # if not os.path.exists(args.save_pr_shelve):
        #     os.mkdir(args.save_pr_shelve)
        w = shelve.open(args.save_pr_shelve)
        # with shelve.open(os.path.abspath(args.save_pr_shelve)) as f:
        w['res'] = res_dict
        w.close()


if __name__ == "__main__":
    # Get current path to set default folders
    # currentPath = os.path.dirname(os.path.abspath(__file__))

    VERSION = '0.1 (beta)'

    parser = argparse.ArgumentParser(
        prog='Object Detection Metrics - Pascal VOC',
        description='This project applies the most popular metrics used to evaluate object detection '
                    'algorithms.\nThe current implemention runs the Pascal VOC metrics.\nFor further references, '
                    'please check:\nhttps://github.com/rafaelpadilla/Object-Detection-Metrics',
        epilog="Developed by: Rafael Padilla (rafael.padilla@smt.ufrj.br)")
    # formatter_class=RawTextHelpFormatter)
    parser.add_argument('-v', '--version', action='version', version='%(prog)s ' + VERSION)
    # Positional arguments
    # Mandatory
    parser.add_argument(
        '-gt',
        '--gtfolder',
        dest='gtFolder',
        # default="/data-private/nas/pspace/4582data0522/VOCdevkitParking4582st20ex2-15crop192-364-556/cal_pr_gt/",  # groundtruths文件夹位置，默认为pr_groundtruths
        default="/data-private/nas/pspace/4582data0522/VOCdevkitParking4582ex2-30crop256-300-556/cal_pr_gt/",  # groundtruths文件夹位置，默认为pr_groundtruths
        # default="/data-private/nas/pspace/4582data0522/VOCdevkitParking4582ex2-20sted-crop256/cal_pr_gt/",  # groundtruths文件夹位置，默认为pr_groundtruths
        # default="/data-private/nas/pspace/4582data0522/VOCdevkitParking4582ex2-15crop192-364-556/cal_pr_gt/",  # groundtruths文件夹位置，默认为pr_groundtruths
        metavar='',
        help='folder containing your ground truth bounding boxes')
    parser.add_argument(
        '-det',
        '--detfolder',
        dest='detFolder',
        # default="/home/chencq/caffe-jacinto-models/scripts/training/voc0712-512x512/mobiledetnet-0.5/voc0712-512x512_mobiledetnet-0.5_20190605_00-13/initial/120000",  # 预测结果label文件夹位置，默认为pr_detections
        # default="/home/chencq/caffe-jacinto-models/scripts/training/voc0712-512x512/mobiledetnet-0.5/voc0712-512x512_mobiledetnet-0.5_20190605_00-21/initial/120000",  # 预测结果label文件夹位置，默认为pr_detections
        # default="/home/chencq/caffe-jacinto-models/scripts/training/voc0712-512x512/mobiledetnet-0.5/0605_00-21_del_channel_0611/initial/120000",  # 预测结果label文件夹位置，默认为pr_detections
        # default="/home/chencq/caffe-jacinto-models/scripts/training/voc0712-512x512/mobiledetnet-0.25/voc0712-512x512_mobiledetnet-0.25_20190612_00-01/initial/120000",  # 预测结果label文件夹位置，默认为pr_detections
        # default="/home/chencq/caffe-jacinto-models/scripts/training/voc0712-256x256/mobiledetnet-0.25/voc0712-256x256_mobiledetnet-0.25_20190612_15-11/initial/120000",  # 预测结果label文件夹位置，默认为pr_detections
        # default="/home/chencq/caffe-jacinto-models/scripts/training/voc0712-256x256/mobiledetnet-0.25/voc0712-256x256_mobiledetnet-0.25_20190612_15-46/initial/120000",  # 预测结果label文件夹位置，默认为pr_detections
        # default="/home/chencq/caffe-jacinto-models/scripts/training/voc0712-256x256/mobiledetnet-0.25/voc0712-256x256_mobiledetnet-0.25_20190612_16-08/initial/120000",  # 预测结果label文件夹位置，默认为pr_detections
        # default="/home/chencq/caffe-jacinto-models/scripts/training/voc0712-512x512/mobiledetnetv2-0.25/voc0712-512x512_mobiledetnetv2-0.25_20190613_20-36/initial/120000",  # 预测结果label文件夹位置，默认为pr_detections
        # default="/home/chencq/caffe-jacinto-models/scripts/training/voc0712-512x512/mobiledetnetv2-0.25/voc0712-512x512_mobiledetnetv2-0.25_20190613_20-23/initial/120000",  # 预测结果label文件夹位置，默认为pr_detections
        # default="/home/chencq/caffe-jacinto-models/scripts/training/voc0712-512x512/mobiledetnet-0.5/voc0712-512x512_mobiledetnet-0.5_20190605_00-28/initial/120000",  # 预测结果label文件夹位置，默认为pr_detections
        # default="/home/chencq/caffe-jacinto-models/scripts/training/voc0712/JDetNet/20190605_00-41_ds_PSP_dsFac_32_hdDS8_1/initial/initial/120000",  # 预测结果label文件夹位置，默认为pr_detections
        # default="/home/chencq/caffe-jacinto-models/scripts/training/voc0712-512x512/mobiledetnet-0.5/voc0712-512x512_mobiledetnet-0.5_20190606_00-48/initial/120000",  # 预测结果label文件夹位置，默认为pr_detections
        # default="/home/chencq/caffe-jacinto-models/scripts/training/voc0712-1280x256/mobiledetnet-0.5/voc0712-1280x256_mobiledetnet-0.5_20190610_17-13/initial/120000",  # 预测结果label文件夹位置，默认为pr_detections
        # default="/home/chencq/caffe-jacinto-models/scripts/training/voc0712-512x512/mobiledetnet-0.25/voc0712-512x512_mobiledetnet-0.25_20190613_17-50/initial/120000",  # 预测结果label文件夹位置，默认为pr_detections
        # default="/home/chencq/caffe-jacinto-models/scripts/training/voc0712-512x512/mobiledetnetv2-1.0/voc0712-512x512_mobiledetnetv2-1.0_20190613_18-20/initial/120000",  # 预测结果label文件夹位置，默认为pr_detections
        # default="/home/chencq/caffe-jacinto-models/scripts/training/voc0712-512x512/mobiledetnetv2-0.5/voc0712-512x512_mobiledetnetv2-0.5_20190614_17-10/initial/120000",  # 预测结果label文件夹位置，默认为pr_detections
        # default="/home/chencq/caffe-jacinto-models/scripts/training/voc0712-512x512/mobiledetnetv2-1.0/voc0712-512x512_mobiledetnetv2-1.0_20190615_00-38/initial/120000",  # 预测结果label文件夹位置，默认为pr_detections
        # default="/home/chencq/caffe-jacinto-models/scripts/training/voc0712-512x512/mobiledetnetv2-0.5/voc0712-512x512_mobiledetnetv2-0.5_20190615_14-12/initial/120000",  # 预测结果label文件夹位置，默认为pr_detections
        # default="/home/chencq/caffe-jacinto-models/scripts/training/voc0712-512x512/mobiledetnetv2-0.25/voc0712-512x512_mobiledetnetv2-0.25_20190615_14-15/initial/120000",  # 预测结果label文件夹位置，默认为pr_detections
        # default="/home/chencq/caffe-jacinto-models/scripts/training/voc0712-512x512/mobiledetnetv2-0.25/voc0712-512x512_mobiledetnetv2-0.25_20190615_14-30/initial/120000",  # 预测结果label文件夹位置，默认为pr_detections
        # default="/home/chencq/caffe-jacinto-models/scripts/training/voc0712-512x512/mobiledetnetv2-0.25/voc0712-512x512_mobiledetnetv2-0.25_20190615_14-41/initial/120000",  # 预测结果label文件夹位置，默认为pr_detections
        # default="/home/chencq/caffe-jacinto-models/scripts/training/voc0712-512x512/mobiledetnet-0.5/voc0712-512x512_mobiledetnet-0.5_20190617_16-34/initial/120000",  # 预测结果label文件夹位置，默认为pr_detections
        # default="/home/chencq/caffe-jacinto-models/scripts/training/voc0712-512x512/mobiledetnet-0.5/voc0712-512x512_mobiledetnet-0.5_20190617_20-06/initial/120000",  # 预测结果label文件夹位置，默认为pr_detections
        # default="/home/chencq/caffe-jacinto-models/scripts/training/voc0712-512x512/mobiledetnet-0.5/voc0712-512x512_mobiledetnet-0.5_20190617_20-22/initial/120000",  # 预测结果label文件夹位置，默认为pr_detections
        # default="/home/chencq/caffe-jacinto-models/scripts/training/voc0712-512x512/mobiledetnet-0.5/voc0712-512x512_mobiledetnet-0.5_20190618_10-56/initial/120000",  # 预测结果label文件夹位置，默认为pr_detections
        # default="/home/chencq/caffe-jacinto-models/scripts/training/voc0712-512x512/mobiledetnet-0.5/voc0712-512x512_mobiledetnet-0.5_20190605_00-21/initial/90000",  # 预测结果label文件夹位置，默认为pr_detections
        # default="/home/chencq/caffe-jacinto-models/scripts/training/voc0712-512x512/mobiledetnet-0.5/0605_00-21_del_channel_0611/initial/68000",  # 预测结果label文件夹位置，默认为pr_detections
        # default="/home/chencq/caffe-jacinto-models/scripts/training/voc0712-512x512/mobiledetnet-0.5/voc0712-512x512_mobiledetnet-0.5_20190618_11-09/initial/48000",  # 预测结果label文件夹位置，默认为pr_detections
        # default="/home/chencq/caffe-jacinto-models/scripts/training/voc0712-512x512/mobiledetnet-0.5/voc0712-512x512_mobiledetnet-0.5_20190618_23-25/initial/120000",  # 预测结果label文件夹位置，默认为pr_detections
        # default="/home/chencq/caffe-jacinto-models/scripts/training/voc0712-512x512/mobiledetnet-0.25/voc0712-512x512_mobiledetnet-0.25_20190618_23-44/initial/150000",  # 预测结果label文件夹位置，默认为pr_detections
        # default="/home/chencq/caffe-jacinto-models/scripts/training/voc0712-512x512/mobiledetnet-0.5/voc0712-512x512_mobiledetnet-0.5_20190618_15-02/initial/18000",  # 预测结果label文件夹位置，默认为pr_detections
        # default="/home/chencq/caffe-jacinto-models/scripts/training/voc0712-512x512/mobiledetnet-0.5/voc0712-512x512_mobiledetnet-0.5_20190620_17-33/initial/32000",  # 预测结果label文件夹位置，默认为pr_detections
        # default="/home/chencq/caffe-jacinto-models/scripts/training/voc0712-512x512/mobiledetnet-0.5/voc0712-512x512_mobiledetnet-0.5_20190620_18-37/initial/48000",  # 预测结果label文件夹位置，默认为pr_detections
        # default="/home/chencq/caffe-jacinto-models/scripts/training/voc0712-512x512/mobiledetnet-0.5/voc0712-512x512_mobiledetnet-0.5_20190621_12-00/initial/52000",  # 预测结果label文件夹位置，默认为pr_detections
        # default="/home/chencq/caffe-jacinto-models/scripts/training/voc0712-512x512/mobiledetnet-0.5/voc0712-512x512_mobiledetnet-0.5_20190624_16-39/initial/102000",  # 预测结果label文件夹位置，默认为pr_detections
        # default="/home/chencq/caffe-jacinto-models/scripts/training/voc0712-512x512/mobiledetnet-0.5/voc0712-512x512_mobiledetnet-0.5_20190624_23-20/initial/74000",  # 预测结果label文件夹位置，默认为pr_detections
        # default="/home/chencq/caffe-jacinto-models/scripts/training/voc0712-512x512/mobiledetnet-0.5/voc0712-512x512_mobiledetnet-0.5_20190625_17-37/initial/78000",  # 预测结果label文件夹位置，默认为pr_detections
        # default="/home/chencq/caffe-jacinto-models/scripts/training/voc0712-512x512/mobiledetnet-0.5/voc0712-512x512_mobiledetnet-0.5_20190625_17-37/initial/70000",  # 预测结果label文件夹位置，默认为pr_detections
        # default="/data-private/nas/pspace/tiPytorchFile/mobilenetv1_voc-lite-0.5-07032330/1000",  #ap: 0.7018 预测结果label文件夹位置，默认为pr_detections
        # default="/home/chencq/caffe-jacinto-models/scripts/training/voc0712-512x512/mobiledetnet-0.5/voc0712-512x512_mobiledetnet-0.5_20190703_11-06/initial/78000",  # 预测结果label文件夹位置，默认为pr_detections
        # default="/home/chencq/caffe-jacinto-models/scripts/training/voc0712-512x512/mobiledetnet-0.5/voc0712-512x512_mobiledetnet-0.5_20190709_17-03/initial/58000",  # 预测结果label文件夹位置，默认为pr_detections
        # default="/home/chencq/caffe-jacinto-models/scripts/training/voc0712-512x512/mobiledetnet-0.5/voc0712-512x512_mobiledetnet-0.5_20190709_15-49/initial/62000",  # 预测结果label文件夹位置，默认为pr_detections
        # default="/home/chencq/caffe-jacinto-models/scripts/training/voc0712-512x512/mobiledetnet-0.5/voc0712-512x512_mobiledetnet-0.5_20190710_15-34/initial/90000",  # 预测结果label文件夹位置，默认为pr_detections
        # default="/home/chencq/caffe-jacinto-models/scripts/training/voc0712-512x512/mobiledetnet-0.5/voc0712-512x512_mobiledetnet-0.5_20190710_15-34/initial/102000",  # 预测结果label文件夹位置，默认为pr_detections
        # default="/home/chencq/caffe-jacinto-models/scripts/training/voc0712-512x512/mobiledetnet-0.5/voc0712-512x512_mobiledetnet-0.5_20190710_15-34/initial/150000",  # 预测结果label文件夹位置，默认为pr_detections
        # default="/home/chencq/caffe-jacinto-models/scripts/training/voc0712-512x512/mobiledetnet-0.5/voc0712-512x512_mobiledetnet-0.5_20190618_10-56/initial/120000",  # 预测结果label文件夹位置，默认为pr_detections
        # default="/home/chencq/caffe-jacinto-models/scripts/training/voc0712-512x512/mobiledetnet-0.5/voc0712-512x512_mobiledetnet-0.5_20190710_15-21/initial/70000",  # 预测结果label文件夹位置，默认为pr_detections
        # default="/home/chencq/caffe-jacinto-models/scripts/training/voc0712-512x512/mobiledetnet-0.5/voc0712-512x512_mobiledetnet-0.5_20190618_15-02/initial/88000",  # 预测结果label文件夹位置，默认为pr_detections
        # default="/data-private/nas/pspace/tiPytorchFile/mobilenetv1_voc-lite-0.5-07032330/1000",  #ap: 0.7018  预测结果label文件夹位置，默认为pr_detections
        default="/data-private/nas/pspace/tiPytorchFile/yolov3_mbv1-100-20190719_0002/100",  #ap: 0.8649 预测结果label文件夹位置，默认为pr_detections
        # default="/home/chencq/caffe-jacinto-models/scripts/training/voc0712-512x512/mobiledetnet-0.5/voc0712-512x512_mobiledetnet-0.5_20190717_15-08/initial/84000",  # 预测结果label文件夹位置，默认为pr_detections
        # default="/home/chencq/caffe-jacinto-models/scripts/training/voc0712-512x512/mobiledetnet-0.5/voc0712-512x512_mobiledetnet-0.5_20190717_15-09/initial/72000",  # 预测结果label文件夹位置，默认为pr_detections
        # default="/home/chencq/caffe-jacinto-models/scripts/training/voc0712-512x512/mobiledetnet-0.5/voc0712-512x512_mobiledetnet-0.5_20190717_21-53/initial/62000",  # 预测结果label文件夹位置，默认为pr_detections
        # default="/home/chencq/caffe-jacinto-models/scripts/training/voc0712-512x512/mobiledetnet-0.5/voc0712-512x512_mobiledetnet-0.5_20190717_22-09/initial/74000",  # 预测结果label文件夹位置，默认为pr_detections
        # default="/home/chencq/caffe-jacinto-models/scripts/training/voc0712-512x512/mobiledetnet-0.5/voc0712-512x512_mobiledetnet-0.5_20190717_22-12/initial/90000",  # 预测结果label文件夹位置，默认为pr_detections
        # default="/home/chencq/caffe-jacinto-models/scripts/training/voc0712-512x512/mobiledetnet-0.5/voc0712-512x512_mobiledetnet-0.5_20190717_22-51/initial/78000",  # 预测结果label文件夹位置，默认为pr_detections
        # default="/home/chencq/caffe-jacinto-models/scripts/training/voc0712-512x512/mobiledetnet-0.5/voc0712-512x512_mobiledetnet-0.5_20190717_22-54/initial/84000",  # 预测结果label文件夹位置，默认为pr_detections
        metavar='',
        help='folder containing your detected bounding boxes')
    parser.add_argument(
        '-method',
        '--M',
        dest='method',
        default=MethodAveragePrecision.EveryPointInterpolation,
        # default=MethodAveragePrecision.ElevenPointInterpolation,  # 预测结果label文件夹位置，默认为pr_detections
        # default=os.path.join(currentPath, 'prediction_label'),
        metavar='',
        help='Interpolation method')

    # Optional
    parser.add_argument(
        '-t',
        '--threshold',
        dest='iouThreshold',
        type=float,
        default=0.5,  # iouThreshold iou阈值，默认为0.5
        metavar='',
        help='IOU threshold. Default 0.5')
    parser.add_argument(
        '-gtformat',
        dest='gtFormat',
        metavar='',
        default='xyrb',  # 标签format， 默认为xyrb
        help='format of the coordinates of the ground truth bounding boxes: '
             '(\'xywh\': <left> <top> <width> <height>)'
             ' or (\'xyrb\': <left> <top> <right> <bottom>)')
    parser.add_argument(
        '-detformat',
        dest='detFormat',
        metavar='',
        default='xyrb',  # 预测结果format， 默认为xyrb
        help='format of the coordinates of the detected bounding boxes '
             '(\'xywh\': <left> <top> <width> <height>) '
             'or (\'xyrb\': <left> <top> <right> <bottom>)')
    parser.add_argument(
        '-gtcoords',
        dest='gtCoordinates',
        default='abs',  # 标签坐标属性， 默认为绝对坐标abs
        metavar='',
        help='reference of the ground truth bounding box coordinates: absolute '
             'values (\'abs\') or relative to its image size (\'rel\')')
    parser.add_argument(
        '-detcoords',
        default='abs',  # 预测结果属性， 默认为绝对坐标abs
        dest='detCoordinates',
        metavar='',
        help='reference of the ground truth bounding box coordinates: '
             'absolute values (\'abs\') or relative to its image size (\'rel\')')
    parser.add_argument(
        '-imgsize',  # 图像尺寸， 当坐标为相对坐标时需要
        dest='imgSize',
        metavar='',
        help='image size. Required if -gtcoords or -detcoords are \'rel\'')
    parser.add_argument(
        '-sp', '--savepath',
        dest='savePath',
        default=None,  # plot保存位置文件夹，默认为results文件夹
        metavar='',
        help='folder where the plots are saved')
    parser.add_argument(
        '-np',
        '--noplot',
        dest='showPlot',
        default=False,  # 是否显示plot, 默认为False
        action='store_false',
        help='no plot is shown during execution')
    parser.add_argument(
        '--save_pr_shelve',
        dest='save_pr_shelve',
        default='',  # 保存 precision recall 为shelve 的路径，若为空，则不保存
        action='store_false',
        help='save precision recall as shelve')
    args = parser.parse_args()
    print(args.detFolder)
    print(args.gtFolder)
    # os.system("python3 -V")
    process_pr_curve(args)

