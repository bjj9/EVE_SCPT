from __future__ import print_function
import os
import pickle
import random
import copy
from ipdb import set_trace as st

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv

class StNet(nn.Module):
    def __init__(self):
        super(StNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

        # Spatial transformer localization-network
        self.localization = nn.Sequential(
            nn.Conv2d(2, 8, kernel_size=7),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),

            nn.Conv2d(8, 12, kernel_size=5),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),

            nn.Conv2d(12, 16, kernel_size=5),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
        )

        # Regressor for the 3 * 2 affine matrix
        self.fc_loc = nn.Sequential(
            nn.Linear(16 * 5 * 12, 32),
            nn.ReLU(True),
            nn.Linear(32, 3 * 2)
        )

        # Initialize the weights/bias with identity transformation
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

        self.sigmoid = nn.Sigmoid()

    # Spatial transformer network forward function
    def stn(self, x):
        # print('before stn', x.size(), '---jjdiief spacial_transform_tutorial.py')
        xs = self.localization(x)

        # st()
        # x = nn.Conv2d(2, 8, kernel_size=7).cuda()(x)
        # x = nn.MaxPool2d(2, stride=2).cuda()(x)
        # x = nn.ReLU(True).cuda()(x)
        #
        # x = nn.Conv2d(8, 12, kernel_size=5).cuda()(x)
        # x = nn.MaxPool2d(2, stride=2).cuda()(x)
        # x = nn.ReLU(True).cuda()(x)
        #
        # x = nn.Conv2d(12, 16, kernel_size=5).cuda()(x)
        # x = nn.MaxPool2d(2, stride=2).cuda()(x)
        # x = nn.ReLU(True).cuda()(x)

        # print(count_parameters(self.localization), count_parameters(self.fc_loc))
        # print('after stn', xs.size(), '---rgffdfg spacial_transform_tutorial.py')
        # raise

        xs = xs.view(-1, 16 * 5 * 12)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)

        grid = F.affine_grid(theta, x.size())
        x = F.grid_sample(x, grid)

        return x

    # def get_affine_theta(self, x):
    #     xs = self.localization(x)
    #     # print(count_parameters(self.localization), count_parameters(self.fc_loc))
    #     # print('after stn', xs.size(), '---rgffdfg spacial_transform_tutorial.py')
    #     # raise
    #     xs = xs.view(-1, 16 * 5 * 12)
    #     theta = self.fc_loc(xs)
    #     theta = theta.view(-1, 2, 3)
    #     return theta

    def forward(self, x):
        # transform the input

        x = self.stn(x)
        heatmap = self.sigmoid(x) # convert to [0, 1] to be compared with ground truth heatmap


        #print(x.shape, '---ewfjoijdf')
        #raise


        # Perform the usual forward pass
        # x = F.relu(F.max_pool2d(self.conv1(x), 2))
        # x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        # x = x.view(-1, 320)
        # x = F.relu(self.fc1(x))
        # x = F.dropout(x, training=self.training)
        # x = self.fc2(x)
        return heatmap

class StRefine(object):
    def __init__(self, StNet_path, stnet_heatmap_size=(128, 72), full_screen_size=(1920, 1080), heatmap_blur=(15,15)):
        if StNet_path is None:
            self.StNet = None
        else:
            self.StNet = StNet().cuda()
            self.StNet.load_state_dict(torch.load(StNet_path))
            print('StRefine loaded from', StNet_path, '---jfejoij st_refine_net.py')

        self.full_screen_size = full_screen_size
        self.stnet_heatmap_size = stnet_heatmap_size
        self.heatmap_guassian_blur = heatmap_blur
        self.softargmax_xs = None
        self.softargmax_ys = None

    def refine(self, PoG_px, PoG_history, return_full=False):
        ''' PoG_pxs, PoG_histories are type np array'''
        # 1. get combmap  in torch
        # try:
        combmap, is_valid = self.creat_combmap(PoG_px.detach().cpu().numpy(), PoG_history.detach().cpu().numpy())
        # except:
        #     st()

        if not is_valid:
            if not return_full:
                return copy.deepcopy(PoG_px), is_valid
            else:
                return copy.deepcopy(PoG_px), is_valid, None, None

        combmap_tensor = torch.from_numpy(combmap).cuda()

        # 2. run stnet
        c, h, w = combmap_tensor.shape
        output_combmap_tensor = self.StNet(combmap_tensor.reshape(-1, c, h, w))

        # 3. retreive refined gaze
        #st()
        n, _, h, w = output_combmap_tensor.shape
        gaze_heatmap = output_combmap_tensor[:, 0, :, :]
        PoG_pxs_refined = self.soft_argmax(gaze_heatmap.reshape(n, 1, h, w))

        # 4. record refined PoGs conditioned on validity
        PoG_px_refined = PoG_pxs_refined[0]

        if not return_full:
            return PoG_px_refined, is_valid
        else:
            return PoG_px_refined, is_valid, combmap, output_combmap_tensor.detach().cpu().numpy()

    # level 1 methods for self.refine()
    def creat_combmap(self, PoG_px, PoG_history):
        in_screen = lambda xy: (0 <= xy[0] <= self.full_screen_size[0]) and (0 <= xy[1] <= self.full_screen_size[1])
        is_valid = in_screen(PoG_px)
        #st()
        if is_valid:
            combmap = self.create_gaze_and_history_combmap(PoG_px, PoG_history)
            combmap = self.numpy_arr_normalised(combmap)
            return combmap, is_valid
        else:
            return None, is_valid

    def numpy_arr_normalised(self, arr, mean=0.5, std=0.5):
        arr = np.transpose(arr.astype(np.float32), (2, 0, 1)) # convert to float32, move from H x W x C to C x H x W
        arr = (arr - mean)/std
        return arr

    def numpy_arr_2_normalised_tensor(self, arr, mean=0.5, std=0.5):
        arr = self.numpy_arr_normalised(arr, mean, std)
        tens = torch.from_numpy(arr)
        return tens
    def soft_argmax(self, heatmaps):
        if self.softargmax_xs is None:
            # Assume normalized coordinate [0, 1] for numeric stability
            w, h = self.stnet_heatmap_size
            ref_xs, ref_ys = np.meshgrid(np.linspace(0, 1.0, num=w, endpoint=True),
                                         np.linspace(0, 1.0, num=h, endpoint=True),
                                         indexing='xy')
            ref_xs = np.reshape(ref_xs, [1, h*w])
            ref_ys = np.reshape(ref_ys, [1, h*w])
            # softargmax_xs = torch.tensor(ref_xs.astype(np.float32)).to(device)
            # softargmax_ys = torch.tensor(ref_ys.astype(np.float32)).to(device)
            self.softargmax_xs = torch.tensor(ref_xs.astype(np.float32)).cuda()
            self.softargmax_ys = torch.tensor(ref_ys.astype(np.float32)).cuda()
        ref_xs, ref_ys = self.softargmax_xs, self.softargmax_ys

        # Yield softmax+integrated coordinates in [0, 1]
        n, _, h, w = heatmaps.shape
        assert(w == self.stnet_heatmap_size[0])
        assert(h == self.stnet_heatmap_size[1])
        beta = 1e2
        x = heatmaps.view(-1, h*w)
        x = F.softmax(beta * x, dim=-1)

        #print(ref_xs.shape, ref_ys.shape, x.shape, '---ejfioji')

        lmrk_xs = torch.sum(ref_xs * x, dim=-1)
        lmrk_ys = torch.sum(ref_ys * x, dim=-1)

        # Return to actual coordinates ranges
        pixel_xs = torch.clamp(self.full_screen_size[0] * lmrk_xs, 0.0, self.full_screen_size[0])
        pixel_ys = torch.clamp(self.full_screen_size[1] * lmrk_ys, 0.0, self.full_screen_size[1])
        return torch.stack([pixel_xs, pixel_ys], dim=-1)

    # level 2 methods for self.creat_combmap()
    def create_gaze_and_history_combmap(self, PoG_px, PoG_history):
        gazemap = self.create_gaze_heatmap(PoG_px, sigma=10.0, gaze_heatmap_size=self.stnet_heatmap_size,
                                           actual_screen_size=self.full_screen_size,
                                           guassian_blur=self.heatmap_guassian_blur)
        trajmap = self.create_history_gaze_path_map(PoG_history, history_trajectory_map_size=self.stnet_heatmap_size,
                                                    actual_screen_size=self.full_screen_size,
                                                    guassian_blur=self.heatmap_guassian_blur)
        combmap = np.stack([gazemap, trajmap], axis=2)
        #combmap = np.stack([trajmap, ], axis=2)
        return combmap

    # level 3 methods for self.create_gaze_and_history_combmap()
    def create_gaze_heatmap(self, centre, sigma=10.0, gaze_heatmap_size=(256, 144), actual_screen_size=(1920, 1080), guassian_blur=(15,15)):
        #centre, sigma = (1300, 690), 10.0
        #gaze_heatmap_size = 256, 144
        #actual_screen_size = 1920, 1080

        w, h = 256, 144 # initial heatmap size
        xs = np.arange(0, w, step=1, dtype=np.float32)
        ys = np.expand_dims(np.arange(0, h, step=1, dtype=np.float32), -1)
        heatmap_xs = xs
        heatmap_ys = ys
        #heatmap_xs = torch.tensor(xs).cuda()
        #heatmap_ys = torch.tensor(ys).cuda()

        heatmap_alpha = -0.5 / (sigma ** 2)
        cx = (w / actual_screen_size[0]) * centre[0]
        cy = (h / actual_screen_size[1]) * centre[1]
        #st()
        heatmap = np.exp(heatmap_alpha * ((heatmap_xs - cx)**2 + (heatmap_ys - cy)**2))
        heatmap = cv.GaussianBlur(heatmap, guassian_blur, 3)
        if (w, h) != gaze_heatmap_size:
            heatmap = cv.resize(heatmap, gaze_heatmap_size)
        heatmap = self.normalise_arr(heatmap)
        #heatmap = 1e-8 + heatmap  # Make the zeros non-zero (remove collapsing issue)
        # heatmap_on = torch.tensor(heatmap).cuda()
        # heatmap.unsqueeze(0)
        # plt.imshow(heatmap, origin='upper')
        # plt.show()
        # heatmap.shape
        return heatmap
    def create_history_gaze_path_map(self, PoG_pxs, history_trajectory_map_size=(256, 144), actual_screen_size=(1920, 1080), guassian_blur=(15,15)):
        #xys = sample['PoG_history_gt'][sample['PoG_history_gt_validity']]
        xys = PoG_pxs
        #history_trajectory_map_size = 256, 144
        #actual_screen_size = 1920, 1080
        w, h = 256, 144

        trajmap = np.zeros((h, w))
        xys_copy = copy.deepcopy(xys)

        xys_copy[:, 0] *= (w / actual_screen_size[0])
        xys_copy[:, 1] *= (h / actual_screen_size[1])
        arrPt = np.array(xys_copy, np.int32).reshape((-1, 1, 2))

        trajmap = cv.polylines(trajmap, [arrPt], isClosed=False, color=(1.0,), thickness=2)
        trajmap = cv.GaussianBlur(trajmap, guassian_blur, 3)
        if (w, h) != history_trajectory_map_size:
            trajmap = cv.resize(trajmap, history_trajectory_map_size)
        trajmap = self.normalise_arr(trajmap)
        # plt.imshow(trajmap, origin='upper')
        # plt.show()
        # heatmap.shape
        return trajmap

    # level 4 method for self.create_gaze_heatmap() and self.create_history_gaze_path_map()
    def normalise_arr(self, arr):
        mmax, mmin = np.max(arr), np.min(arr)
        assert mmax > mmin
        arr = (arr - mmin +1e-8)/(mmax - mmin + 2e-8)
        return arr

if __name__ == "__main__":
    model_dir = '/samba/room/storage/Bji/network_outputs/spacial_transform/evec_transform/model_fake_eyetracking_dataset_1kg_128_72_error00_valid_random_ii_in_sequence_person_gt'
    state_name = 'spatical_transform_model_fake_eyetracking_dataset_1kg_128_72_error00_valid_random_ii_in_sequence_person_gt_lr_0.1_100_full'
    state_path = model_dir + '/' + state_name
    st_refine = StRefine(StNet_path=state_path)


