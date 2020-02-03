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

from lib.calpr.prlib.BoundingBox import BoundingBox
from lib.calpr.prlib.BoundingBoxes import BoundingBoxes
from lib.calpr.prlib.Evaluator import *
from lib.calpr.prlib.utils import BBFormat
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
    iouThreshold = args.iouThreshold
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

    # f = open(os.path.join(savePath, 'results.txt'), 'w')
    # f.write('Object Detection Metrics\n')
    # f.write('https://github.com/rafaelpadilla/Object-Detection-Metrics\n\n\n')
    # f.write('Average Precision (AP), Precision and Recall per class:')

    # each detection is a class
    res_dict = {}
    for metricsPerClass in detections:
        cl = metricsPerClass['class']
        ap = metricsPerClass['AP']
        res_dict[cl] = ap
    return res_dict

class PRargs:
    def __init__(self,
            detFolder="/data-private/nas/pspace/tiPytorchFile/yolov3_mbv1-100-20190719_0002/100",
            gtFolder="/data-private/nas/pspace/4582data0522/VOCdevkitParking4582ex2-30crop256-300-556/cal_pr_gt/"):
        self.gtFolder = gtFolder
        self.detFolder = detFolder
        self.method = MethodAveragePrecision.EveryPointInterpolation
        self.iouThreshold=0.5
        self.gtFormat = 'xyrb'
        self.detFormat = 'xyrb'
        self.gtCoordinates = 'abs'
        self.detCoordinates = 'abs'
        self.imgSize = ''
        self.savePath = None
        self.showPlot = False
        self.save_pr_shelve = ''
if __name__ == "__main__":

    input = PRargs()
    print(input.detFolder)
    print(input.gtFolder)
    # os.system("python3 -V")
    res_dict = process_pr_curve(input)
    print(res_dict)

