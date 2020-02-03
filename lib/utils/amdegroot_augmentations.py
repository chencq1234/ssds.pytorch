import torch
from torchvision import transforms
import cv2
import numpy as np
import types
from numpy import random


def intersect(box_a, box_b):
    # try:
    max_xy = np.minimum(box_a[:, 2:], box_b[2:])
    min_xy = np.maximum(box_a[:, :2], box_b[:2])
    inter = np.clip((max_xy - min_xy), a_min=0, a_max=np.inf)
    # except Exception as exp:
    #     print(exp)
    return inter[:, 0] * inter[:, 1]


def jaccard_numpy(box_a, box_b):
    """Compute the jaccard overlap of two sets of boxes.  The jaccard overlap
    is simply the intersection over union of two boxes.
    E.g.:
        A ∩ B / A ∪ B = A ∩ B / (area(A) + area(B) - A ∩ B)
    Args:
        box_a: Multiple bounding boxes, Shape: [num_boxes,4]
        box_b: Single bounding box, Shape: [4]
    Return:
        jaccard overlap: Shape: [box_a.shape[0], box_a.shape[1]]
    """
    inter = intersect(box_a, box_b)
    area_a = ((box_a[:, 2]-box_a[:, 0]) *
              (box_a[:, 3]-box_a[:, 1]))  # [A,B]
    area_b = ((box_b[2]-box_b[0]) *
              (box_b[3]-box_b[1]))  # [A,B]
    union = area_a + area_b - inter
    return inter / union  # [A,B]


class Compose(object):
    """Composes several augmentations together.
    Args:
        transforms (List[Transform]): list of transforms to compose.
    Example:
        >>> augmentations.Compose([
        >>>     transforms.CenterCrop(10),
        >>>     transforms.ToTensor(),
        >>> ])
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, boxes=None, labels=None, kp=None):
        for t in self.transforms:
            img, boxes, labels, kp = t(img, boxes, labels, kp)
        return img, boxes, labels


class Lambda(object):
    """Applies a lambda as a transform."""

    def __init__(self, lambd):
        assert isinstance(lambd, types.LambdaType)
        self.lambd = lambd

    def __call__(self, img, boxes=None, labels=None, kps=None):
        return self.lambd(img, boxes, labels, kps)


class ConvertFromInts(object):
    def __call__(self, image, boxes=None, labels=None, kps=None):
        return image.astype(np.float32), boxes, labels, kps


class SubtractMeans(object):
    def __init__(self, mean):
        self.mean = np.array(mean, dtype=np.float32)

    def __call__(self, image, boxes=None, labels=None, kp=None):
        image = image.astype(np.float32)
        image -= self.mean
        if len(boxes) == 0:
            boxes = np.zeros((1, 4))
            labels = np.array([0])
        return image.astype(np.float32), boxes, labels, kp


class ToAbsoluteCoords(object):
    def __call__(self, image, boxes=None, labels=None, kp=None):
        height, width, channels = image.shape
        boxes[:, 0] *= width
        boxes[:, 2] *= width
        boxes[:, 1] *= height
        boxes[:, 3] *= height
        kp[:, :, 0] *= width
        kp[:, :, 1] *= height
        return image, boxes, labels, kp


class ToPercentCoords(object):
    def __call__(self, image, boxes=None, labels=None, kp=None):
        height, width, channels = image.shape
        # print("shape:",boxes.shape)
        boxes[:, 0] /= width
        boxes[:, 2] /= width
        boxes[:, 1] /= height
        boxes[:, 3] /= height
        kp[:, :, 0] /= width
        kp[:, :, 1] /= height
        return image, boxes, labels, kp


class Resize(object):
    def __init__(self, size=None):
        if size is None:
            size = [416, 416]
        self.size = size

    def __call__(self, image, boxes=None, labels=None, kp=None):
        interp_methods = [cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_NEAREST, cv2.INTER_LANCZOS4]
        interp_method = interp_methods[random.randint(5)]
        image = cv2.resize(image, (self.size[0],
                                 self.size[0]), interpolation=interp_method)
        return image, boxes, labels, kp


class RandomSaturation(object):
    def __init__(self, lower=0.5, upper=1.5):
        self.lower = lower
        self.upper = upper
        assert self.upper >= self.lower, "contrast upper must be >= lower."
        assert self.lower >= 0, "contrast lower must be non-negative."

    def __call__(self, image, boxes=None, labels=None, kp=None):
        if random.randint(2):
            image[:, :, 1] *= random.uniform(self.lower, self.upper)

        return image, boxes, labels, kp


class RandomHue(object):
    def __init__(self, delta=18.0):
        assert delta >= 0.0 and delta <= 360.0
        self.delta = delta

    def __call__(self, image, boxes=None, labels=None, kp=None):
        if random.randint(2):
            image[:, :, 0] += random.uniform(-self.delta, self.delta)
            image[:, :, 0][image[:, :, 0] > 360.0] -= 360.0
            image[:, :, 0][image[:, :, 0] < 0.0] += 360.0
        return image, boxes, labels, kp


class RandomLightingNoise(object):
    def __init__(self):
        self.perms = ((0, 1, 2), (0, 2, 1),
                      (1, 0, 2), (1, 2, 0),
                      (2, 0, 1), (2, 1, 0))

    def __call__(self, image, boxes=None, labels=None, kp=None):
        if random.randint(2):
            swap = self.perms[random.randint(len(self.perms))]
            shuffle = SwapChannels(swap)  # shuffle channels
            image = shuffle(image)
        return image, boxes, labels, kp


class ConvertColor(object):
    def __init__(self, current='BGR', transform='HSV'):
        self.transform = transform
        self.current = current

    def __call__(self, image, boxes=None, labels=None, kp=None):
        if self.current == 'BGR' and self.transform == 'HSV':
            image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        elif self.current == 'HSV' and self.transform == 'BGR':
            image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
        else:
            raise NotImplementedError
        return image, boxes, labels, kp


class RandomContrast(object):
    def __init__(self, lower=0.5, upper=1.5):
        self.lower = lower
        self.upper = upper
        assert self.upper >= self.lower, "contrast upper must be >= lower."
        assert self.lower >= 0, "contrast lower must be non-negative."

    # expects float image
    def __call__(self, image, boxes=None, labels=None, kp=None):
        if random.randint(2):
            alpha = random.uniform(self.lower, self.upper)
            image *= alpha
        return image, boxes, labels, kp


class RandomBrightness(object):
    def __init__(self, delta=32):
        assert delta >= 0.0
        assert delta <= 255.0
        self.delta = delta

    def __call__(self, image, boxes=None, labels=None, kp=None):
        if random.randint(2):
            delta = random.uniform(-self.delta, self.delta)
            image += delta
        return image, boxes, labels, kp


class ToCV2Image(object):
    def __call__(self, tensor, boxes=None, labels=None, kp=None):
        return tensor.cpu().numpy().astype(np.float32).transpose((1, 2, 0)), boxes, labels, kp


class ToTensor(object):
    def __call__(self, cvimage, boxes=None, labels=None, kp=None):
        return torch.from_numpy(cvimage.astype(np.float32)).permute(2, 0, 1), boxes, labels, kp


class RandomSampleCrop(object):
    """Crop
    Arguments:
        img (Image): the image being input during training
        boxes (Tensor): the original bounding boxes in pt form
        labels (Tensor): the class labels for each bbox
        mode (float tuple): the min and max jaccard overlaps
    Return:
        (img, boxes, classes)
            img (Image): the cropped image
            boxes (Tensor): the adjusted bounding boxes in pt form
            labels (Tensor): the class labels for each bbox
    """
    def __init__(self):
        self.sample_options = (
            # using entire original input image
            None,
            # sample a patch s.t. MIN jaccard w/ obj in .1,.3,.4,.7,.9
            (0.1, None),
            (0.3, None),
            (0.7, None),
            (0.9, None),
            # randomly sample a patch
            (None, None),
        )

    def __call__(self, image, boxes=None, labels=None, kp=None):
        height, width, _ = image.shape
        while True:
            # randomly choose a mode
            mode = random.choice(self.sample_options)
            if mode is None:
                return image, boxes, labels, kp

            min_iou, max_iou = mode
            if min_iou is None:
                min_iou = float('-inf')
            if max_iou is None:
                max_iou = float('inf')

            # max trails (50)
            for _ in range(50):
                current_image = image

                w = random.uniform(0.3 * width, width)
                h = random.uniform(0.3 * height, height)

                # aspect ratio constraint b/t .5 & 2
                if h / w < 0.5 or h / w > 2:
                    continue

                left = random.uniform(width - w)
                top = random.uniform(height - h)

                # convert to integer rect x1,y1,x2,y2
                rect = np.array([int(left), int(top), int(left+w), int(top+h)])

                # calculate IoU (jaccard overlap) b/t the cropped and gt boxes
                overlap = jaccard_numpy(boxes, rect)

                # is min and max overlap constraint satisfied? if not try again
                if overlap.min() < min_iou and max_iou < overlap.max():
                    continue

                # cut the crop from the image
                current_image = current_image[rect[1]:rect[3], rect[0]:rect[2],
                                              :]

                # keep overlap with gt box IF center in sampled patch
                centers = (boxes[:, :2] + boxes[:, 2:]) / 2.0

                # mask in all gt boxes that above and to the left of centers
                m1 = (rect[0] < centers[:, 0]) * (rect[1] < centers[:, 1])

                # mask in all gt boxes that under and to the right of centers
                m2 = (rect[2] > centers[:, 0]) * (rect[3] > centers[:, 1])

                # mask in that both m1 and m2 are true
                mask = m1 * m2

                # have any valid boxes? try again if not
                if not mask.any():
                    continue

                # take only matching gt boxes
                current_boxes = boxes[mask, :].copy()

                # take only matching gt labels
                current_labels = labels[mask]

                # should we use the box left and top corner or the crop's
                current_boxes[:, :2] = np.maximum(current_boxes[:, :2],
                                                  rect[:2])
                # adjust to crop (by substracting crop's left,top)
                current_boxes[:, :2] -= rect[:2]

                current_boxes[:, 2:] = np.minimum(current_boxes[:, 2:],
                                                  rect[2:])
                # adjust to crop (by substracting crop's left,top)
                current_boxes[:, 2:] -= rect[:2]

                return current_image, current_boxes, current_labels, kp


def genMatrixScaleRotate(image, rotateDegree):
    x = (image.shape[1] - 1) / 2
    y = (image.shape[0] - 1) / 2
    scaleFactor = 1.0
    matrixScaleRotate = np.eye(3)
    matrixScaleRotate[:2] = cv2.getRotationMatrix2D(center=(x, y), angle=rotateDegree, scale=scaleFactor)
    return matrixScaleRotate


def genMatrixTranslate(image, matrix):
    vertices = np.array([0, 0, image.shape[1] - 1, 0, image.shape[1] - 1, image.shape[0] - 1, 0, image.shape[0] - 1], dtype=np.float)
    vertices = vertices.reshape(1, -1, 2)
    vertices = cv2.perspectiveTransform(vertices, matrix).reshape(-1, 2)
    minX, minY = np.floor(vertices.min(0)).astype(np.int)
    maxX, maxY = np.ceil(vertices.max(0)).astype(np.int)
    matrixTranslate = np.eye(3)
    matrixTranslate[0, 2] -= minX
    matrixTranslate[1, 2] -= minY
    return matrixTranslate, (maxX - minX + 1), (maxY - minY + 1)


def transformImage(image, matrix):
    mat_translate, width, height = genMatrixTranslate(image, matrix)
    M = mat_translate @ matrix
    newImage = cv2.warpPerspective(image, M, dsize=(width, height), flags=cv2.INTER_CUBIC, borderValue=(104, 117, 123))
    return newImage, M


def transformVertices(vertices, M):
    coords = vertices.reshape(1, -1, 2)
    coords = cv2.perspectiveTransform(coords, M)
    coords = coords.reshape(-1, 4, 2)
    # vertices= coords.reshape(-1, 8)
    return coords


class RandomRotate(object):
    def __init__(self, angle):
        self.angle = angle

    def __call__(self, image, boxes=None, labels=None, kps=None):
        if random.randint(3) > 2:
            return image, boxes, labels, kps
        angle = np.random.randint(-self.angle, self.angle)
        matrixScaleRotate = genMatrixScaleRotate(image, angle)
        image_new, M = transformImage(image, matrixScaleRotate)
        kps = transformVertices(kps, M)
        boxes_new = []
        for kp in kps:
            boxes_new.append([kp[:, 0].min(), kp[:, 1].min(), kp[:, 0].max(), kp[:, 1].max()])
        # boxes_new = np.array([
        #     list(map(min, kps[:, :, 0])),
        #     list(map(min, kps[:, :, 1])),
        #     list(map(max, kps[:, :, 0])),
        #     list(map(max, kps[:, :, 1]))])
        return image_new, np.array(boxes_new), labels, kps

class Expand(object):
    def __init__(self, mean):
        self.mean = mean

    def __call__(self, image, boxes, labels, kps=None):
        if random.randint(2):
            return image, boxes, labels, kps

        height, width, depth = image.shape
        ratio = random.uniform(1, 4)
        left = random.uniform(0, width*ratio - width)
        top = random.uniform(0, height*ratio - height)

        expand_image = np.zeros(
            (int(height*ratio), int(width*ratio), depth),
            dtype=image.dtype)
        expand_image[:, :, :] = self.mean
        expand_image[int(top):int(top + height),
                     int(left):int(left + width)] = image
        image = expand_image

        boxes = boxes.copy()
        # try:
        boxes[:, :2] += (int(left), int(top))
        boxes[:, 2:] += (int(left), int(top))
        # except Exception as exp:
        #     print(exp)
        return image, boxes, labels, kps


class RandomMirror(object):
    def __call__(self, image, boxes, classes, kps=None):
        _, width, _ = image.shape
        if random.randint(2):
            image = image[:, ::-1]
            boxes = boxes.copy()
            boxes[:, 0::2] = width - boxes[:, 2::-2]
        return image, boxes, classes, kps


class SwapChannels(object):
    """Transforms a tensorized image by swapping the channels in the order
     specified in the swap tuple.
    Args:
        swaps (int triple): final order of channels
            eg: (2, 1, 0)
    """

    def __init__(self, swaps):
        self.swaps = swaps

    def __call__(self, image):
        """
        Args:
            image (Tensor): image tensor to be transformed
        Return:
            a tensor with channels swapped according to swap
        """
        # if torch.is_tensor(image):
        #     image = image.data.cpu().numpy()
        # else:
        #     image = np.array(image)
        image = image[:, :, self.swaps]
        return image


class PhotometricDistort(object):
    def __init__(self):
        self.pd = [
            RandomContrast(),
            ConvertColor(transform='HSV'),
            RandomSaturation(),
            RandomHue(),
            ConvertColor(current='HSV', transform='BGR'),
            RandomContrast()
        ]
        self.rand_brightness = RandomBrightness()
        self.rand_light_noise = RandomLightingNoise()

    def __call__(self, image, boxes, labels, kps):
        im = image.copy()
        im, boxes, labels, kps = self.rand_brightness(im, boxes, labels, kps)
        if random.randint(2):
            distort = Compose(self.pd[:-1])
        else:
            distort = Compose(self.pd[1:])
        im, boxes, labels = distort(im, boxes, labels, kps)
        return self.rand_light_noise(im, boxes, labels, kps)


class SSDAugmentation(object):
    def __init__(self, size=None, mean=(104, 117, 123)):
        if size is None:
            size = [300, 300]
        self.mean = mean
        self.size = size
        self.augment = Compose([
            ConvertFromInts(),
            ToAbsoluteCoords(),
            PhotometricDistort(),
            RandomRotate(3),
            Expand(self.mean),
            RandomSampleCrop(),
            RandomMirror(),
            ToPercentCoords(),
            Resize(self.size),
            SubtractMeans(self.mean)
        ])

    def __call__(self, img, boxes, labels, kp):
        try:
            kp = kp[:, :, :2]
        except Exception as exp:
            print(exp)
        return self.augment(img, boxes, labels, kp)
