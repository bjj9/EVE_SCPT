"""Copyright 2021 Hangzhou Dianzi University, Jun Bao

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
import logging

import os
# import datetime
# import time
# import matplotlib.pyplot as plt


import numpy as np
import torch
from torch import nn
from torchvision.models.resnet import BasicBlock, ResNet

from core import DefaultConfig
#from ipdb import set_trace as st

config = DefaultConfig()
logger = logging.getLogger(__name__)

# device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
curr_device_index = torch.cuda.current_device()
#device = torch.device('cuda:'+str(curr_device_index)  if torch.cuda.is_available() else "cpu")
device = torch.device(("cuda:0" if config.use_one_gpu is None else 'cuda:'+str(config.use_one_gpu) )  if torch.cuda.is_available() else "cpu")
device = torch.device('cuda:0')
print(device, '---rgefedf eye_net.py')
half_pi = 0.5 * np.pi


class EyeNet(nn.Module):
    def __init__(self):
        super(EyeNet, self).__init__()

        num_features = (
            config.eye_net_rnn_num_features
            if config.eye_net_use_rnn
            else config.eye_net_static_num_features
        )

        # CNN backbone (ResNet-18 with instance normalization)
        self.cnn_layers = ResNet(block=BasicBlock, layers=[2, 2, 2, 2],
                                 num_classes=num_features,
                                 norm_layer=nn.InstanceNorm2d)
        self.fc_common = nn.Sequential(
            nn.Linear(num_features + (2 if config.eye_net_use_head_pose_input else 0),
                      num_features),
            nn.SELU(inplace=True),
            nn.Linear(num_features, num_features),
        )

        if config.eye_net_use_rnn:
            # Define RNN cell
            rnn_cells = []
            for i in range(config.eye_net_rnn_num_cells):
                if config.eye_net_rnn_type == 'RNN':
                    rnn_cells.append(nn.RNNCell(input_size=config.eye_net_rnn_num_features,
                                                hidden_size=config.eye_net_rnn_num_features))
                elif config.eye_net_rnn_type == 'LSTM':
                    rnn_cells.append(nn.LSTMCell(input_size=config.eye_net_rnn_num_features,
                                                 hidden_size=config.eye_net_rnn_num_features))
                elif config.eye_net_rnn_type == 'GRU':
                    rnn_cells.append(nn.GRUCell(input_size=config.eye_net_rnn_num_features,
                                                hidden_size=config.eye_net_rnn_num_features))
                else:
                    raise ValueError('Unknown RNN type for EyeNet: %s' % config.eye_net_rnn_type)
            self.rnn_cells = nn.ModuleList(rnn_cells)
        else:
            self.static_fc = nn.Sequential(
                nn.Linear(num_features, num_features),
                nn.SELU(inplace=True),
            )

        # FC layers
        self.fc_to_gaze = nn.Sequential(
            nn.Linear(num_features, num_features),
            nn.SELU(inplace=True),
            nn.Linear(num_features, 2, bias=False),
            nn.Tanh(),
        )
        self.fc_to_pupil = nn.Sequential(
            nn.Linear(num_features, num_features),
            nn.SELU(inplace=True),
            nn.Linear(num_features, 1),
            nn.ReLU(inplace=True),
        )

        # self.fc_to_validity = nn.Sequential(
        #     nn.Linear(num_features, num_features),
        #     nn.SELU(inplace=True),
        #     nn.Linear(num_features, 1),
        #     nn.Tanh(),
        # )

        # Set gaze layer weights to zero as otherwise this can
        # explode early in training
        nn.init.zeros_(self.fc_to_gaze[-2].weight)

        # self.forward_kwargs = {'side': None}
    def forward(self, input_dict, output_dict=None, previous_output_dict=None, side=None):
        # ee0 = time.time()
        # Pick input image
        if (side + '_eye_patch') in output_dict:
            input_image = output_dict[side + '_eye_patch']
        else:
            input_image = input_dict[side + '_eye_patch']

        # ee1 = time.time()
        # Compute CNN features
        initial_features = self.cnn_layers(input_image)

        # ee2 = time.time()
        # Process head pose input if asked for
        if config.eye_net_use_head_pose_input:
            initial_features = torch.cat([initial_features, input_dict[side + '_h']], axis=1)
        initial_features = self.fc_common(initial_features)

        # ee3 = time.time()
        # Apply RNN cells
        if config.eye_net_use_rnn:

            rnn_features = initial_features
            for i, rnn_cell in enumerate(self.rnn_cells):
                # em0 = time.time()
                suffix = '_%d' % i

                # Retrieve previous hidden/cell states if any
                previous_states = None
                if previous_output_dict is not None:
                    previous_states = previous_output_dict[side + '_eye_rnn_states' + suffix]

                # em1 = time.time()
                # Inference through RNN cell
                states = rnn_cell(rnn_features, previous_states)
                # em2 = time.time()

                # Decide what the output is and store back current states
                if isinstance(states, tuple):
                    rnn_features = states[0]
                    output_dict[side + '_eye_rnn_states' + suffix] = states
                else:
                    rnn_features = states
                    output_dict[side + '_eye_rnn_states' + suffix] = states
                # em3 = time.time()
                # timeMarks = np.array([em0, em1, em2, em3,])
                # timePeriodsStr = ['em' + str(i) + '.em' + str(i + 1) for i in range(len(timeMarks) - 1)]
                # timePeriods = timeMarks[1:] - timeMarks[:-1]
                # if i == 0:
                #     print('eye_net rnn cell', list(zip(timePeriodsStr, timePeriods)), '---bfefddd eye_net.py')
            features = rnn_features
        else:
            features = self.static_fc(initial_features)
        # ee4 = time.time()

        # Final prediction
        #print('rnn_features size:', features.shape, 'initial_features size:', initial_features.shape, '---jjijijfe eye_net')
        #raise
        gaze_prediction = half_pi * self.fc_to_gaze(features)
        pupil_size = self.fc_to_pupil(features)
        #tracking_validity = torch.sigmoid(self.fc_to_validity(initial_features)) # use features from single frame to estimate validity
        # ee5 = time.time()

        # For gaze, the range of output values are limited by a tanh and scaling
        output_dict[side + '_g_initial'] = gaze_prediction

        # Estimate of pupil size
        output_dict[side + '_pupil_size'] = pupil_size.reshape(-1)

        # Estimate of tracking validity
        #output_dict[side + '_tracking_validity'] = tracking_validity.reshape(-1)
        # ee6 = time.time()
        # If network frozen, we're gonna detach gradients here
        if config.eye_net_frozen:
            output_dict[side + '_g_initial'] = output_dict[side + '_g_initial'].detach()
        # ee7 = time.time()
        # timeMarks = np.array([ee0, ee1, ee2, ee3, ee4, ee5, ee6, ee7])
        # timePeriodsStr = ['ee' + str(i) + '.ee' + str(i + 1) for i in range(len(timeMarks) - 1)]
        # timePeriods = timeMarks[1:] - timeMarks[:-1]
        # print('eye_net forward', list(zip(timePeriodsStr, timePeriods)), 'input image size', input_image.size(), '---fjefjoijijd eye_net.py')





