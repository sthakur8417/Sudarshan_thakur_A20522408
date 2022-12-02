import glob
import os
import torch
import cv2
import numpy as np

import imgaug as ia
import imgaug.augmenters as iaa
from imgaug.augmentables import Keypoint, KeypointsOnImage
import matplotlib.pyplot as plt
import imageio


class Normalize(object):
    """Convert a color image to grayscale and normalize the color range to [-1,1]."""



    def __call__(self, sample):
        image, key_pts = sample['image'], sample['keypoints']
        for i in range(image.shape[0]):
            image[i,:,:] = (image[i,:,:] -0.5) / 0.5

        return {'image': image, 'keypoints': key_pts}


class Rescale(object):
    """Rescale the image in a sample to a given size.e

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):

        image, key_pts = sample['image'], sample['keypoints']




        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        img = cv2.resize(image, (new_w, new_h))



        key_pts = key_pts * [new_w / w, new_h / h]

        return {'image': img, 'keypoints': key_pts}


class Aug(object):

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        image, key_pts = sample['image'], sample['keypoints']

        kps = KeypointsOnImage([
            Keypoint(x=key_pts[0][0], y=key_pts[0][1]),
            Keypoint(x=key_pts[1][0], y=key_pts[1][1]),
            Keypoint(x=key_pts[2][0], y=key_pts[2][1]),
            Keypoint(x=key_pts[3][0], y=key_pts[3][1])
        ], shape=image.shape)

        seq = iaa.Sequential([

            iaa.Fliplr(0.5),
            iaa.Flipud(0.5),
            iaa.Affine(
                rotate=(-30, 30),
                scale=(0.75, 1.25)


        ])


        image_aug, kps_aug = seq(image=image, keypoints=kps)

        key_pts_aug = [[0, 0] for i in range(4)]
        for i in range(len(kps.keypoints)):
            before = kps.keypoints[i]
            after = kps_aug.keypoints[i]


            key_pts_aug[i][0] = after.x
            key_pts_aug[i][1] = after.y

        key_pts_aug = np.array(key_pts_aug)






        return {'image': image_aug, 'keypoints': key_pts_aug}





class RandomCrop(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        image, key_pts = sample['image'], sample['keypoints']

        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        image = image[top: top + new_h,
                left: left + new_w]

        key_pts = key_pts - [left, top]

        return {'image': image,  'keypoints': key_pts}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors【0，1】"""

    def __call__(self, sample):
        image, key_pts = sample['image'], sample['keypoints']


        if (len(image.shape) == 2):

            image = image.reshape(image.shape[0], image.shape[1], 1)




        image = image.transpose((2, 0, 1))

        gt = torch.from_numpy(key_pts)
        gt2 = torch.from_numpy(key_pts)
        return {'image': torch.from_numpy(image),

                'keypoints': torch.from_numpy(key_pts)}




def getoriimage(sample):
        image = sample.numpy()
        for i in range(image.shape[0]):
            image[i,:,:] = (image[i,:,:] + 1) * 0.5

        image = image.transpose((1, 2, 0))
        return image



