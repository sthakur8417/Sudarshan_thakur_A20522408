import glob
import os
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pandas as pd
import cv2
import copy

import torchvision.transforms as transforms
from datapreprocess import Normalize, Rescale, ToTensor, getoriimage,Aug
from utilis.netparameter160 import config

numtest = 0
numtest2 =0
numtest3 = 0
numtest4 = 0
numtest5 = 0

transform_train = transforms.Compose([Rescale((480, 640)),

                                      ToTensor(),
                                      Normalize()])


class MouseKeypointsDataset(Dataset):
    """PDMouse dataset."""

    def __init__(self, csv_file, root_dir, transform=transform_train):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """


        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.key_pts_frame)

    def __getitem__(self, idx):
        image_name = os.path.join(self.root_dir,
                                  self.key_pts_frame.iloc[idx, 0])
        print("image_name--------------",image_name)




        if (image.shape[2] == 4):


        H, W = image.shape[:2]








        gt = np.array(gt)
        gt = gt.astype('float').reshape(-1, 2)






        keypointsnotvisible = np.array(keypointsnotvisible)

        samplenotrans = {'image': image, 'keypoints': gt}

        if self.transform:
            sample = self.transform(samplenotrans)

        gt_trans = sample['keypoints']
        image_trans = sample['image']
        H, W = image_trans.shape[1:3]
        gt_trans_numpy = gt_trans.numpy()


        gt_trans = gt_trans.numpy() * [1/4, 1/4]
        H = H // 4
        W = W // 4
        heatmaps = self._putGaussianMaps(gt_trans, keypointsnotvisible, H, W, 1, config['sigma'] )



        gt_trans2 = gt_trans * [2, 2]
        H = H * 2
        W = W * 2
        heatmaps_2s = self._putGaussianMaps(gt_trans2, keypointsnotvisible, H, W, 1, config['sigma'] )











        sample = {'image': image_trans, 'heatmaps': heatmaps, 'heatmaps_2s': heatmaps_2s, 'keypoints': gt_trans2}


        return sample

    def _putGaussianMap(self, center, visible_flag, crop_size_y, crop_size_x, stride, sigma):
        """
        根据一个中心点,生成一个heatmap
        :param center:
        :return:heapmaps
        """

        grid_x = crop_size_x // stride
        if visible_flag == False:

        start = stride / 2.0 - 0.5
        y_range = [i for i in range(int(grid_y))]
        x_range = [i for i in range(int(grid_x))]
        xx, yy = np.meshgrid(x_range, y_range)
        xx = xx * stride + start
        yy = yy * stride + start
        d2 = (xx - center[0]) ** 2 + (yy - center[1]) ** 2
        exponent = d2 / 2.0 / sigma / sigma

        return heatmap

    def _putGaussianMaps(self, keypoints, keypointsnotvisible, crop_size_y, crop_size_x, stride, sigma):
        """
        计算每个点的heatmaps
        :param keypoints: (15,2)
        :param crop_size_y: int
        :param crop_size_x: int
        :param stride: int
        :param sigma: float
        :return:返回15个heatmaps（15，96，96）
        """
        all_keypoints = keypoints
        notvisible = keypointsnotvisible
        point_num = all_keypoints.shape[0]
        heatmaps_this_img = []
        for k in range(point_num):



            heatmap = self._putGaussianMap(all_keypoints[k], flag, crop_size_y, crop_size_x, stride, sigma)

            heatmaps_this_img.append(heatmap)

        return heatmaps_this_img

    def visualize_heatmap_target(self, oriImg, heatmap, keypoint):
        global numtest2

        plt.figure()
        plt.imshow(oriImg)
        plt.imshow(heatmap, alpha=0.7)

        if (numtest2 < config['test_num']):

            if(keypoint ==3):
                numtest2 = numtest2 + 1

    def visualize_heatmap_target_gt(self, oriImg, heatmap, keypoint):
        global numtest5

        plt.figure()
        plt.imshow(oriImg)
        plt.imshow(heatmap, alpha=0.7)

        if (numtest5 < config['test_num']):
            plt.savefig("results/gt/image%d_%d" % (numtest5, keypoint), bbox_inches='tight', dpi=130,

            if (keypoint == 3):
                numtest5 = numtest5 + 1

    def visualize_heatmap_target_all(self, oriImg, heatmap):
        global numtest3

        plt.figure()
        plt.imshow(oriImg)
        plt.imshow(heatmap[0], alpha=0.4)
        plt.imshow(heatmap[1], alpha=0.4)
        plt.imshow(heatmap[2], alpha=0.4)
        plt.imshow(heatmap[3], alpha=0.4)

        if (numtest3 < config['test_num']):

            numtest3 = numtest3 + 1

    def visualize_heatmap_target_all_gt(self, oriImg, heatmap):
        global numtest4

        plt.figure()
        plt.imshow(oriImg)
        plt.imshow(heatmap[0], alpha=0.4)
        plt.imshow(heatmap[1], alpha=0.4)
        plt.imshow(heatmap[2], alpha=0.4)
        plt.imshow(heatmap[3], alpha=0.4)

        if (numtest4 < config['test_num']):

            numtest4 = numtest4 + 1



    def show_keypoints(self, image, key_pts, gtpoints):
        '''
            """Show image with keypoints"""
        :param image: [C,h,w]
        :param key_pts: [4,2]
        :return:
        '''
        global numtest

        plt.figure()

        plt.imshow(image)
        cValue = ['r', 'y', 'g', 'b']



        plt.plot(key_pts[[0, 1], 0], key_pts[[0, 1], 1], color='m')
        plt.plot(key_pts[[0, 2], 0], key_pts[[0, 2], 1], color='m')
        plt.plot(key_pts[[1, 2], 0], key_pts[[1, 2], 1], color='m')
        plt.plot(key_pts[[1, 3], 0], key_pts[[1, 3], 1], color='m')
        plt.plot(key_pts[[2, 3], 0], key_pts[[2, 3], 1], color='m')
        plt.scatter(key_pts[:, 0], key_pts[:, 1], s=30, marker='.', c=cValue)
        plt.scatter(gtpoints[:, 0], gtpoints[:, 1], s=30, marker='+', linewidths =8, c=cValue)

        a = config['test_num']
        if (numtest < config['test_num']):

            numtest = numtest + 1
            plt.show()








if __name__ == '__main__':

    from utilis.netparameter160 import config

    mouse_dataset = MouseKeypointsDataset(csv_file=config['trainsample'],
                                          root_dir='')

    print('Length of dataset: ', len(mouse_dataset))
    config['test_num'] = len(mouse_dataset)
    dataLoader = DataLoader(dataset=mouse_dataset, batch_size=config['batch_size'], shuffle=False)
    for i, data in enumerate(dataLoader):
        print(data['image'].size())
        print(data['heatmaps'].size())
        print(data['keypoints'].size())





