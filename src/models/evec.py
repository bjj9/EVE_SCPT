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

import os
import time
import datetime
#import matplotlib.pyplot as plt
import pickle
import copy
import random
import numpy as np
import torch
import torch.nn as nn
from ipdb import set_trace as st

from core import DefaultConfig
from losses.cross_entropy import CrossEntropyLoss
from losses.euclidean import EuclideanLoss
from losses.angular import AngularLoss
from losses.mse import MSELoss
from losses.l1 import L1Loss
from utils.load_model import load_weights_for_instance

from .common import (batch_make_heatmaps, batch_make_gaze_history_maps, soft_argmax,
                     to_screen_coordinates, apply_offset_augmentation,
                     calculate_combined_gaze_direction)
from .eye_net import EyeNet
from .refine_net import RefineNet
from .st_refine_net import StRefine


PIXEL_PER_DEGREE = 42.082546   # calculated by averaging the 'metric_euc_PoG_px_intial'/'metric_ang_g_initial' for each clip in eye_net results on validation data
MM_PER_PIXEL = 0.287991895 # average for x and y axis (0.28802082 and 0.28796297 respectively)
CM_PER_PIXEL_X, CM_PER_PIXEL_Y = 0.28802082/10, 0.28796297/10
CM_PER_DEGREE = 1.211943216896467 # calculated by PIXEL_PER_ANGLE * 1.0 * MM_PER_PIXEL / 10.0
MARGIN_CM_ONE_DEGREE = CM_PER_DEGREE
SCREEN_W_CM, SCREEN_H_CM = 55.3000, 31.1000
SCREEN_W, SCREEN_H = 1920, 1080
SCREEN_CENTER_X, SCREEN_CENTER_Y = 893.99963, 520.8466  # PoG_px ground truth average in training data,
                                                        # after data beyond screen removed,
                                                        # 5.94% data removed in total (see appendix in the end)
SCREEN_CENTER_CM_X, SCREEN_CENTER_CM_Y = 25.751211, 14.9969 # PoG_cm ground truth average in training data,
                                                            # after data beyond screen removed,
                                                            # 6.18% data removed in total (see appendix in the end)


config = DefaultConfig()
# device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
curr_device_index = torch.cuda.current_device()
#device = torch.device('cuda:'+str(curr_device_index)  if torch.cuda.is_available() else "cpu")
device = torch.device(("cuda:0" if config.use_one_gpu is None else 'cuda:'+str(config.use_one_gpu) )  if torch.cuda.is_available() else "cpu")
device = torch.device('cuda:0')
print(device, '---nefdfsasd evec.py')

cross_entropy_loss = CrossEntropyLoss()
euclidean_loss = EuclideanLoss()
angular_loss = AngularLoss()
mse_loss = MSELoss()
l1_loss = L1Loss()
bec_loss_without_validity_check = nn.BCELoss()

# for competition 1.95
class EVEC(nn.Module):
    def __init__(self, output_predictions=False, online_refinement_starts_from=2,
                 fixed_history_len=None, input_memory_path='',
                 one_video_to_select=None, one_sample_to_plot=None):
        super(EVEC, self).__init__()
        self.output_predictions = output_predictions

        # 1. Network to estimate gaze direction and pupil size (mm)
        self.eye_net = EyeNet()
        if config.eye_net_load_pretrained:
            if config.eye_net_load_from_specified_location != '':
                state_dict = torch.load(config.eye_net_load_from_specified_location, map_location=device)
                state_dict = dict([(k[8:], v) for k,v in state_dict.items()]) # there is 'eye_net.' infront every key, remove it
                load_weights_for_instance(self.eye_net, state_dict)
                print('eye_net loaded from', config.eye_net_load_from_specified_location, '---nwefji evec.py')
            else:
                load_weights_for_instance(self.eye_net)
                print('eye_net loaded from default url', '---nwefji evec.py')

        # ##################################
        # print(self.eye_net.state_dict()['fc_to_gaze.0.weight'], self.eye_net.state_dict()['fc_to_gaze.0.weight'].size(), '---bwefdfdf evec.py')
        # ############################################################

        if config.eye_net_frozen:
            for param in self.eye_net.parameters():
                param.requires_grad = False

        # 2. Network to refine gaze by spatical tranforming
        state_path = '/samba/room/codespace/pythonWorkingSpace/EVE_SCPT/src/models/trained_model_params/spatical_transform_model_fake_eyetracking_dataset_1kg_128_72_error00_valid_random_ii_in_sequence_person_gt_lr_0.1_99_full.pt'
        self.st_refine_net = StRefine(StNet_path=state_path)
        self.fixed_history_len = fixed_history_len
        if self.fixed_history_len is not None:
            assert input_memory_path != ''
            self.fixed_length_history = self.collect_fixed_length_history(input_memory_path, one_video_to_select=one_video_to_select)
            self.running_memory_version = 'not initiated'
        self.st_transform_enabled = True
        self.write_empty_memories()

        # 3. Network to refine estimated gaze based on:
        #   a) history of point of gaze (PoG)
        #   b) screen content
        self.refine_net = RefineNet() if config.refine_net_enabled else None
        if config.refine_net_enabled and config.refine_net_load_pretrained:
            if config.refine_net_load_from_specified_location != '':
                print('add refine net')
                raise
            else:
                load_weights_for_instance(self.refine_net)

        self.eye_side = config.eye_side # hwiojefoiwjef eve.py
        self.jiugong_enabled = config.jiugong_enabled
        self.memory_in_stable_thresh = 999
        self.memory_center_calculation_out_outlier_thresh = 1
        self.center_drag = 3
        self.online_refinement_starts_from = online_refinement_starts_from # (540, 1081, 2163, 3245) # at least 2 for building history heatmap
        self.count_memory_update = 0

        # plotting
        self.one_sample_to_plot = one_sample_to_plot
        self.plotting_materials_one_sample = {}
        self.frame_count = 0
        self.plotted = False

        # self.forward_kwargs = {'current_epoch': None,  'side': None}

    # __init__ methods
    def collect_fixed_length_history(self, memory_path, one_video_to_select=None):
        if one_video_to_select is not None:
            subject_to_select, cam_pos_to_select, video_name_to_select = one_video_to_select

        subject_memories = pickle.load(open(memory_path, 'rb'))
        memory_pi, memory_bi, memory_pi_star, memory_pi_mean_std, \
        memory_pi_mean_after_outlier_removal, memory_video_id, memory_last_update_at = subject_memories
        his_len = self.fixed_history_len
        if his_len == 'full': his_len = np.inf
        fixed_length_history = {}
        for subject in sorted(memory_pi_star.keys()):
            if one_video_to_select is not None:
                if subject != subject_to_select: continue
            #if subject not in ['val02',]: continue
            fixed_length_history[subject] = {}
            for cam_pos in sorted(memory_pi_star[subject].keys()):
                if one_video_to_select is not None:
                    if cam_pos != cam_pos_to_select: continue
                # prepare memory materials
                ps = np.array(memory_pi[subject][cam_pos])
                bs = np.array(memory_bi[subject][cam_pos])
                vids = np.array([v.split('/')[-1] for v in memory_video_id[subject][cam_pos]])
                all_video_ids = np.unique(vids)
                fixed_length_history[subject][cam_pos] = {}
                for video_id_curr in sorted(all_video_ids):
                    if one_video_to_select is not None:
                        if video_id_curr.split('/')[-1] != video_name_to_select: continue
                    H = {'pi':[], 'bi':[], 'vis': [], 'total':0}
                    videos_random = all_video_ids[:]
                    random.shuffle(videos_random)
                    for vi in videos_random:
                        if vi == video_id_curr: continue
                        H['vis'].append(vi.split('/')[-1][4:7])
                        select = (vids == vi)
                        ps_selected = ps[select]
                        bs_selected = bs[select]
                        H['pi'] = H['pi'] + list(ps_selected)
                        H['bi'] = H['bi'] + list(bs_selected)
                        H['total'] += len(ps_selected)
                        if H['total'] >= his_len:
                            num_to_delete = H['total'] - his_len
                            H['pi'] = H['pi'][:-num_to_delete]
                            H['bi'] = H['bi'][:-num_to_delete]
                            H['total'] -= num_to_delete
                            break
                    assert H['total'] <= his_len # there could be cases when there are not enough frames for this his_len
                    print(subject, cam_pos, video_id_curr, H['vis'], H['total'], '---jjfiejrjjf')
                    fixed_length_history[subject][cam_pos][video_id_curr] = H
                    # plt.plot(np.array(H['pi'])[:, 0], np.array(H['pi'])[:, 1])
                    # plt.show()
        # check_dict(memory_pi_star)
        # check_dict(fixed_length_history)
        # print(one_video_to_select, '---jfejijij evec.py')
        # st()
        return fixed_length_history
    def write_empty_memories(self):
        self.memory_pi = {}
        self.memory_bi = {}
        self.memory_pi_star = {}
        self.memory_pi_mean_std = {}
        self.memory_pi_mean_after_outlier_removal = {}
        self.memory_video_id = {} # for memory augemented results
        self.memory_last_update_at = {}

    def forward(self, full_input_dict, create_images=False, current_epoch=None):
        # tt0 = time.time()
        if self.training:  # pick first source
            assert len(full_input_dict) == 1  # for now say there's 1 training data source
            full_input_dict = next(iter(full_input_dict.values()))

        # There are some labels that we need to calculate ourselves
        # print(device, current_epoch, create_images, '---nbiefiko eve.py')
        self.calculate_additional_labels(full_input_dict, current_epoch=current_epoch)

        # NOTE: In general, parts of the architecture will read from `input_dict`
        #       and write to `output_dict`. However, as we want some modularity,
        #       the routines will always look for previous-step estimates first,
        #       which will exist in `output_dict`.
        #
        #       For example, if the key `left_gaze_origin` exists in `output_dict`,
        #       the following routines will use this estimate instead of the
        #       ground-truth existing in `input_dict`.

        # self.peek_dict(full_input_dict, '---ejijioj')

        intermediate_dicts = []  # One entry per time step in sequence
        initial_heatmap_history = []
        refined_heatmap_history = []

        sequence_len = next(iter(full_input_dict.values())).shape[1]
        # print('jiugong enabled', self.jiugong_enabled, '---nbuhjefi eve.py')
        # tt1 = time.time()
        #memory_visual_center = self.retrieve_memory_visual_center(full_input_dict)

        for t in range(sequence_len):

            # tm0 = time.time()
            # Create new dict formed only of the specific camera view's data
            sub_input_dict = {}
            for k, v in full_input_dict.items():
                if isinstance(v, torch.Tensor):
                    sub_v = v[:, t, :] if v.ndim > 2 else v[:, t]
                    sub_input_dict[k] = sub_v

            # if t == 0:
            #     suffix = '_'+full_input_dict['participant'][0]+'_'+full_input_dict['subfolder'][0]+'_'+full_input_dict['camera'][0]
            #     save_image_to_temp(sub_input_dict['left_eye_patch'][0], 'left_eye_patch'+suffix)
            #     save_image_to_temp(sub_input_dict['right_eye_patch'][0], 'right_eye_patch'+suffix)

            # Step 0) Define output structure that will hold
            #         - intermediate outputs
            #         - final outputs
            #         - individual loss terms
            sub_output_dict = {}

            # Step 1a) From each eye patch, estimate gaze direction and pupil size
            previous_output_dict = (intermediate_dicts[-1] if len(intermediate_dicts) > 0 else None)
            #sub_input_dict['previous_output_dict'] = previous_output_dict

            self.plot_this_sample = False

            subject = full_input_dict['participant'][0]
            video_id = full_input_dict['subfolder'][0]
            cam_pos = full_input_dict['camera'][0]
            if self.one_sample_to_plot is not None:
                sub_to_plot, video_id_to_plot, cam_pos_to_plot, i_to_plot, region_to_plot = self.one_sample_to_plot
                self.plot_this_sample = (subject == sub_to_plot) and (video_id == video_id_to_plot) and \
                                        (cam_pos == cam_pos_to_plot) and (i_to_plot == self.frame_count)
            if self.plot_this_sample:
                self.plotted = True
                self.plotting_materials_one_sample['subject'] = subject
                self.plotting_materials_one_sample['video_id'] = video_id
                self.plotting_materials_one_sample['camera'] = cam_pos
                self.plotting_materials_one_sample['ii'] = i_to_plot
                pass
                #st()

            # tm1 = time.time()
            if config.eye_side in ['left', 'binocular']:
                #self.eye_net.forward_kwargs['side'] = 'left'
                # sub_output_dict_left =  self.eye_net(sub_input_dict, side='left') # nboiwjfoij
                # sub_output_dict = sub_output_dict_left
                self.eye_net(sub_input_dict, output_dict=sub_output_dict, previous_output_dict=previous_output_dict, side='left')  # nboiwjfoij
            # self.peek_dict(sub_output_dict_left, '---nejifj eve.py')
            # tm2 = time.time()
            if config.eye_side in ['right', 'binocular']:
                #self.eye_net.forward_kwargs['side'] = 'right'
                # sub_output_dict_right =  self.eye_net(sub_input_dict, side='right') # nboiwjfoij
                # sub_output_dict = dict(list(sub_output_dict.items()) + list(sub_output_dict_right.items()))
                self.eye_net(sub_input_dict, output_dict=sub_output_dict, previous_output_dict=previous_output_dict, side='right')  # nwfeeeoij

            if self.plot_this_sample:
                self.plotting_materials_one_sample['left_eye_patch'] = sub_input_dict['left_eye_patch'].detach().cpu().numpy()[0]
                self.plotting_materials_one_sample['right_eye_patch'] = sub_input_dict['right_eye_patch'].detach().cpu().numpy()[0]

            # self.peek_dict(sub_output_dict_right, '---jfeifjoij eve.py')
            # self.peek_dict(sub_output_dict, '---jctvybu eve.py')
            # tm3 = time.time()

            # self.eye_net(sub_input_dict, sub_output_dict, side='right',
            #              previous_output_dict=previous_output_dict)
            #st()
            # During training: add random offsets to gaze directions

            if self.training and config.refine_net_do_offset_augmentation:
                #st()
                # eye_list = self.generate_eye_list()
                for side in ['left', 'right']:
                    sub_output_dict[side + '_g_initial_unaugmented'] = sub_output_dict[side + '_g_initial']

                    # print(sub_input_dict[side + '_kappa_fake'], '--w-fijijd eve.py')
                    # print(sub_input_dict['head_R'], '--weeffffjd')
                    # print(sub_output_dict[side + '_g_initial'], '--w-efjoij eve.py')

                    sub_output_dict[side + '_g_initial'] = apply_offset_augmentation(
                        sub_output_dict[side + '_g_initial'],
                        sub_input_dict['head_R'],
                        sub_input_dict[side + '_kappa_fake'],
                    )

                self.from_g_to_PoG_history(full_input_dict, sub_input_dict, sub_output_dict,
                                           input_suffix='initial',
                                           output_suffix='initial_unaugmented',
                                           heatmap_history=None,
                                           gaze_heatmap_sigma=config.gaze_heatmap_sigma_initial,
                                           history_heatmap_sigma=config.gaze_heatmap_sigma_history)

                self.from_g_to_PoG_history(full_input_dict, sub_input_dict, sub_output_dict,
                                           input_suffix='initial',
                                           output_suffix='initial_augmented',
                                           heatmap_history=None,
                                           gaze_heatmap_sigma=config.gaze_heatmap_sigma_initial,
                                           history_heatmap_sigma=config.gaze_heatmap_sigma_history)

            # Step 1b) Estimate PoG, create heatmaps, and heatmap history
            # look for right_POG_cm_initial
            self.from_g_to_PoG_history(full_input_dict, sub_input_dict, sub_output_dict,
                                       input_suffix='initial',
                                       output_suffix='initial',
                                       heatmap_history=initial_heatmap_history,
                                       gaze_heatmap_sigma=config.gaze_heatmap_sigma_initial,
                                       history_heatmap_sigma=config.gaze_heatmap_sigma_history)

            # Step 2) Digest screen content frame
            if self.refine_net:
                self.refine_net(sub_input_dict, sub_output_dict, previous_output_dict=previous_output_dict)

                # Step 2b) Update refined heatmap history
                refined_heatmap_history.append(sub_output_dict['heatmap_final'])
                refined_gaze_history_maps = batch_make_gaze_history_maps(
                    full_input_dict['timestamps'], refined_heatmap_history,
                    full_input_dict['PoG_px_tobii_validity'],
                ) if 'PoG_px_tobii' in full_input_dict else None

                # Step 3) Yield refined final PoG estimate(s)
                sub_output_dict['PoG_px_final'] = soft_argmax(sub_output_dict['heatmap_final'])
                cm_per_px = 0.1 * sub_input_dict['millimeters_per_pixel']
                sub_output_dict['PoG_cm_final'] = torch.mul(
                    sub_output_dict['PoG_px_final'], cm_per_px,
                )

                # and gaze direction '---bwejijjd eve.py'
                sub_output_dict['g_final'] = calculate_combined_gaze_direction(
                    sub_input_dict['o'],
                    10.0 * sub_output_dict['PoG_cm_final'],
                    sub_input_dict['left_R'],  # by definition, 'left_R' == 'right_R'
                    sub_input_dict['camera_transformation'],
                    )
            else:
                if not self.training: # do memory center correction during inference only
                    # if there is no refine net, we still want to have a final result entry in output dict
                    sub_output_dict['PoG_px_final'] = sub_output_dict['PoG_px_initial']
                    cm_per_px = 0.1 * sub_input_dict['millimeters_per_pixel']
                    sub_output_dict['PoG_cm_final'] = torch.mul(sub_output_dict['PoG_px_final'], cm_per_px)
                    sub_output_dict['g_final'] = calculate_combined_gaze_direction(
                        sub_input_dict['o'],
                        10.0 * sub_output_dict['PoG_cm_final'],
                        sub_input_dict['left_R'],  # by definition, 'left_R' == 'right_R'
                        sub_input_dict['camera_transformation'],
                        )
                    # if config.central_calibration_enabled:
                    #     sub_output_dict['predicted_tracking_validity_final'] = sub_output_dict['predicted_tracking_validity_initial']
                    #print('g_final added')
                    #raise

            # Store back outputs
            intermediate_dicts.append(sub_output_dict)
            # tm4 = time.time()
            # timeMarks = np.array([tm0, tm1, tm2, tm3, tm4,])
            # timePeriodsStr = ['tm' + str(i) + '.tm' + str(i + 1) for i in range(len(timeMarks) - 1)]
            # timePeriods = timeMarks[1:] - timeMarks[:-1]
            # if t == 45:
            #     print('seqence', t, list(zip(timePeriodsStr, timePeriods)), '---fekfokekof eve.py')

            self.frame_count += 1






        # tt2 = time.time()
        # Merge intermediate outputs over time steps to yield BxTxF tensors
        full_intermediate_dict = {}
        for k in intermediate_dicts[0].keys():
            sample = intermediate_dicts[0][k]
            if not isinstance(sample, torch.Tensor):
                continue
            full_intermediate_dict[k] = torch.stack([
                intermediate_dicts[i][k] for i in range(sequence_len)
            ], axis=1)
        # print(self.peek_dict(full_intermediate_dict), '---hhuefij eve.py')

        # Copy over some values that we want to yield as NN output
        output_dict = {}
        for k in full_intermediate_dict.keys():
            if k.startswith('output_'):
                output_dict[k] = full_intermediate_dict[k]
        # print(self.peek_dict(output_dict), '---wfefddfsdf eve.py')

        if self.eye_side in ['left', 'binocular']:
            output_dict['left_pupil_size'] = full_intermediate_dict['left_pupil_size']
            if config.eye_validity_net_enabled:
                output_dict['left_tracking_validity'] = full_intermediate_dict['left_tracking_validity']
        if self.eye_side in ['right', 'binocular']:
            output_dict['right_pupil_size'] = full_intermediate_dict['right_pupil_size']
            if config.eye_validity_net_enabled:
                output_dict['right_tracking_validity'] = full_intermediate_dict['right_tracking_validity']

        if config.load_full_frame_for_visualization:
            # Copy over some values manually
            if 'left_g_tobii' in full_input_dict:
                output_dict['left_g_gt'] = full_input_dict['left_g_tobii']
            if 'right_g_tobii' in full_input_dict:
                output_dict['right_g_gt'] = full_input_dict['right_g_tobii']

            output_dict['PoG_px_gt'] = full_input_dict['PoG_px_tobii']
            # print('lenth of PoG_px_gt', len(output_dict['PoG_px_gt']), '---nbjjfjef eye.py')
            output_dict['PoG_px_gt_validity'] = full_input_dict['PoG_px_tobii_validity']

            if self.eye_side in ['left', 'binocular']:
                output_dict['left_g_initial'] = full_intermediate_dict['left_g_initial']
            if self.eye_side in ['right', 'binocular']:
                output_dict['right_g_initial'] = full_intermediate_dict['right_g_initial']
            output_dict['PoG_px_initial'] = full_intermediate_dict['PoG_px_initial']
            if config.refine_net_enabled:
                output_dict['g_final'] = full_intermediate_dict['g_final']
                output_dict['PoG_px_final'] = full_intermediate_dict['PoG_px_final']

        if self.output_predictions:
            output_dict['timestamps'] = full_input_dict['timestamps']
            output_dict['o'] = full_input_dict['o']
            output_dict['left_R'] = full_input_dict['left_R']
            output_dict['head_R'] = full_input_dict['head_R']
            output_dict['g_initial'] = full_intermediate_dict['g_initial']
            output_dict['PoG_px_initial'] = full_intermediate_dict['PoG_px_initial']
            output_dict['PoG_cm_initial'] = full_intermediate_dict['PoG_cm_initial']
            output_dict['left_PoG_cm_initial'] = full_intermediate_dict['left_PoG_cm_initial']
            output_dict['right_PoG_cm_initial'] = full_intermediate_dict['right_PoG_cm_initial']

            if config.central_calibration_enabled:
                output_dict['PoG_px_initial_for_memory'] = full_intermediate_dict['PoG_px_initial_for_memory']
                output_dict['PoG_cm_initial_for_memory'] = full_intermediate_dict['PoG_cm_initial_for_memory']
            output_dict['millimeters_per_pixel'] = full_input_dict['millimeters_per_pixel']
            output_dict['pixels_per_millimeter'] = full_input_dict['pixels_per_millimeter']
            output_dict['camera_transformation'] = full_input_dict['camera_transformation']
            output_dict['inv_camera_transformation'] = full_input_dict['inv_camera_transformation']

            if self.refine_net:
                output_dict['g_final'] = full_intermediate_dict['g_final']
                output_dict['PoG_px_final'] = full_intermediate_dict['PoG_px_final']
                output_dict['PoG_cm_final'] = full_intermediate_dict['PoG_cm_final']

            if (not self.training) and config.central_calibration_enabled: # do central calibration only during inference
                #output_dict['predicted_tracking_validity_final'] = full_intermediate_dict['predicted_tracking_validity_final']
                output_dict['g_final'] = full_intermediate_dict['g_final']
                output_dict['PoG_px_final'] = full_intermediate_dict['PoG_px_final']
                output_dict['PoG_cm_final'] = full_intermediate_dict['PoG_cm_final']

            # Ground-truth related data
            if 'g' in full_input_dict:
                output_dict['g'] = full_input_dict['g']
                output_dict['validity'] = full_input_dict['PoG_px_tobii_validity']
                output_dict['PoG_cm'] = full_input_dict['PoG_cm_tobii']
                output_dict['PoG_px'] = full_input_dict['PoG_px_tobii']


        # Calculate all loss terms and metrics (scores)
        self.calculate_losses_and_metrics(full_input_dict, full_intermediate_dict, output_dict)

        # Calculate the final combined (and weighted) loss
        if not config.multi_gpu:
            # full_loss = torch.zeros(()).to(device)
            full_loss = torch.zeros(()).cuda()
        else:
            full_loss = torch.zeros(()).cuda()

        # Add all losses for the eye network
        #if self.eye_side == 'binocular':
        if 'loss_ang_left_g_initial' in output_dict:
            full_loss += config.loss_coeff_g_ang_initial * (
                    output_dict['loss_ang_left_g_initial'] +
                    output_dict['loss_ang_right_g_initial']
            )
        if 'loss_mse_left_PoG_cm_initial' in output_dict \
                and config.loss_coeff_PoG_cm_initial > 0.0:
            full_loss += config.loss_coeff_PoG_cm_initial * (
                    output_dict['loss_mse_left_PoG_cm_initial'] +
                    output_dict['loss_mse_right_PoG_cm_initial']
            )
        if 'loss_l1_left_pupil_size' in output_dict:
            full_loss += config.loss_coeff_pupil_size * (
                    output_dict['loss_l1_left_pupil_size'] +
                    output_dict['loss_l1_right_pupil_size']
            )

        if ('loss_bce_left_tracking_validity' in output_dict):
            full_loss += config.loss_coeff_tracking_validity * (
                    output_dict['loss_bce_left_tracking_validity'] +
                    output_dict['loss_bce_right_tracking_validity']
            )

        # tt3 = time.time()
        # Add all losses for the eye network
        if 'loss_ang_left_g_initial' in output_dict:
            full_loss += config.loss_coeff_g_ang_initial * (
                    output_dict['loss_ang_left_g_initial'] +
                    output_dict['loss_ang_right_g_initial']
            )
        if 'loss_mse_left_PoG_cm_initial' in output_dict \
                and config.loss_coeff_PoG_cm_initial > 0.0:
            full_loss += config.loss_coeff_PoG_cm_initial * (
                    output_dict['loss_mse_left_PoG_cm_initial'] +
                    output_dict['loss_mse_right_PoG_cm_initial']
            )
        if 'loss_l1_left_pupil_size' in output_dict:
            full_loss += config.loss_coeff_pupil_size * (
                    output_dict['loss_l1_left_pupil_size'] +
                    output_dict['loss_l1_right_pupil_size']
            )


        # # Add all losses for the eye network
        # if ('loss_ang_left_g_initial' in output_dict) or ('loss_ang_right_g_initial' in output_dict):
        #     full_loss += config.loss_coeff_g_ang_initial * (
        #         (output_dict['loss_ang_left_g_initial'] if 'loss_ang_left_g_initial' in output_dict else 0) * left_weight +
        #         (output_dict['loss_ang_right_g_initial'] if 'loss_ang_right_g_initial' in output_dict else 0) * right_weight
        #         # output_dict['loss_ang_left_g_initial'] * 2
        #     )
        # if ('loss_mse_left_PoG_cm_initial' in output_dict) or ('loss_mse_right_PoG_cm_initial' in output_dict) \
        #         and config.loss_coeff_PoG_cm_initial > 0.0:
        #     full_loss += config.loss_coeff_PoG_cm_initial * (
        #             (output_dict['loss_mse_left_PoG_cm_initial'] if 'loss_mse_left_PoG_cm_initial' in output_dict else 0) * left_weight +
        #             (output_dict['loss_mse_right_PoG_cm_initial'] if 'loss_mse_right_PoG_cm_initial' in output_dict else 0) * right_weight
        #         # output_dict['loss_mse_right_PoG_cm_initial']
        #     )
        # if ('loss_l1_left_pupil_size' in output_dict) or ('loss_l1_right_pupil_size' in output_dict):
        #     full_loss += config.loss_coeff_pupil_size * (
        #             (output_dict['loss_l1_left_pupil_size'] if 'loss_l1_left_pupil_size' in output_dict else 0) * left_weight +
        #             (output_dict['loss_l1_right_pupil_size'] if 'loss_l1_right_pupil_size' in output_dict else 0) * right_weight
        #         # output_dict['loss_l1_right_pupil_size']
        #     )

        # Add all losses for the GazeRefineNet
        if 'loss_mse_PoG_cm_final' in output_dict:
            full_loss += config.loss_coeff_PoG_cm_final * output_dict['loss_mse_PoG_cm_final']
        if 'loss_ce_heatmap_initial' in output_dict:
            full_loss += config.loss_coeff_heatmap_ce_initial * \
                         output_dict['loss_ce_heatmap_initial']
        if 'loss_ce_heatmap_final' in output_dict:
            full_loss += config.loss_coeff_heatmap_ce_final * output_dict['loss_ce_heatmap_final']
        if 'loss_mse_heatmap_final' in output_dict:
            full_loss += config.loss_coeff_heatmap_mse_final * output_dict['loss_mse_heatmap_final']

        # print('full_loss size', full_loss.size(), output_dict['loss_ang_left_g_initial'], '---nbejijijfjf eve.py')
        output_dict['full_loss'] = full_loss

        # Store away tensors for visualization
        # tt4 = time.time()
        if create_images:
            if config.load_full_frame_for_visualization:
                output_dict['both_eye_patch'] = torch.cat([
                    full_input_dict['right_eye_patch'], full_input_dict['left_eye_patch'],
                ], axis=4)
            if config.load_screen_content:
                output_dict['screen_frame'] = full_input_dict['screen_frame'][:, -1, :]
            if 'history_initial' in full_intermediate_dict:
                output_dict['initial_gaze_history'] = full_intermediate_dict['history_initial'][:, -1, :]  # noqa
            if 'heatmap_initial' in full_intermediate_dict:
                output_dict['initial_heatmap'] = full_intermediate_dict['heatmap_initial'][:, -1, :]
            if 'heatmap_final' in full_intermediate_dict:
                output_dict['final_heatmap'] = full_intermediate_dict['heatmap_final'][:, -1, :]
                output_dict['refined_gaze_history'] = refined_gaze_history_maps
            if 'heatmap_final' in full_input_dict:
                output_dict['gt_heatmap'] = full_input_dict['heatmap_final'][:, -1, :]

        # self.peek_dict(output_dict, '---fhfweohih eve.py')
        # tt5 = time.time()
        # timeMarks = np.array([tt0, tt1, tt2, tt3, tt4, tt5,])
        # timePeriodsStr = ['tt' + str(i) + '.tt' + str(i + 1) for i in range(len(timeMarks) - 1)]
        # timePeriods = timeMarks[1:] - timeMarks[:-1]
        # print('seqence', t, list(zip(timePeriodsStr, timePeriods)), '---mnjiiytgyuj eve.py')

        # if 'participant' in full_input_dict.keys():
        participant = full_input_dict['participant'][0]
        subfolder = full_input_dict['subfolder'][0]
        camera = full_input_dict['camera'][0]
        str_prefix = participant + '_' + camera
        # else:
        #     str_prefix = 'unknown participant'

        if config.central_calibration_enabled:
            memory_bi = self.memory_bi[participant][camera]
            #print('memory:', str_prefix, 'valid percent:', '{:.2%}'.format(np.sum(memory_bi)/len(memory_bi) if len(memory_bi) > 0 else 0), '---kkejjood evec.py')
        return output_dict

    # original EVE model methods
    def calculate_losses_and_metrics(self, input_dict, intermediate_dict, output_dict):
        # Initial estimates of gaze direction and PoG
        for side in ('left', 'right'):
            input_key = side + '_g_tobii'
            interm_key = (side + '_g_initial_unaugmented'
                          if self.training and config.refine_net_do_offset_augmentation
                          else side + '_g_initial')
            output_key = side + '_g_initial'
            if interm_key in intermediate_dict and input_key in input_dict:
                output_dict['loss_ang_' + output_key] = angular_loss(
                    intermediate_dict[interm_key], input_key, input_dict,
                )

            input_key = side + '_PoG_cm_tobii'
            interm_key = (side + '_PoG_cm_initial_unaugmented'
                          if self.training and config.refine_net_do_offset_augmentation
                          else side + '_PoG_cm_initial')
            output_key = side + '_PoG_cm_initial'
            if interm_key in intermediate_dict and input_key in input_dict:
                output_dict['loss_mse_' + output_key] = mse_loss(
                    intermediate_dict[interm_key], input_key, input_dict,
                )
                output_dict['metric_euc_' + output_key] = euclidean_loss(
                    intermediate_dict[interm_key], input_key, input_dict,
                )

            input_key = side + '_PoG_tobii'
            interm_key = side + '_PoG_px_initial'
            if interm_key in intermediate_dict and input_key in input_dict:
                output_dict['metric_euc_' + interm_key] = euclidean_loss(
                    intermediate_dict[interm_key], input_key, input_dict,
                )

            # Pupil size in mm
            input_key = side + '_p'
            interm_key = side + '_pupil_size'
            if interm_key in intermediate_dict and input_key in input_dict:
                output_dict['loss_l1_' + interm_key] = l1_loss(
                    intermediate_dict[interm_key], input_key, input_dict,
                )

            # tracking validity [0,1]
            if config.eye_validity_net_enabled:
                input_key = side + '_PoG_tobii_validity'
                interm_key = side + '_tracking_validity'
                #print(intermediate_dict[interm_key].shape, input_dict[input_key].shape, '---jkejoij evec.py')
                # print(bec_loss_without_validity_check(
                #     intermediate_dict[interm_key], input_dict[input_key].type(torch.FloatTensor).cuda(),
                # ), '---evfejmmm evec.py')
                # raise
                if interm_key in intermediate_dict and input_key in input_dict:
                    output_dict['loss_bce_' + interm_key] = bec_loss_without_validity_check(
                        intermediate_dict[interm_key], input_dict[input_key].type(torch.FloatTensor).cuda(),
                    ) # loss_bce_left_tracking_validity
                #print('loss_bce_' + interm_key, intermediate_dict[interm_key], '---ejjjdj evec.py xxxxxxxxxxxxxxxxx' )

            # print(input_dict[side + '_eye_patch'][0], '---kweijouij evec.py')
            # print(intermediate_dict[side + '_PoG_px_initial'], '---dfefdf evec')
            # print(intermediate_dict[side + '_pupil_size'], '---wejfioj evec')
            # print(intermediate_dict[interm_key], input_dict[input_key].type(torch.FloatTensor).cuda(), '---jejjii evec')
            # print('image dimensions', torch.min(input_dict[side + '_eye_patch']), torch.max(input_dict[side + '_eye_patch']), '---efjijii')
            # print('loss_bce_' + interm_key, output_dict['loss_bce_' + interm_key], '---kkjjkk evec')


        # print(self.peek_dict(input_dict), '---jfjeiejj eve.py')
        # print(self.peek_dict(intermediate_dict), '---bjejijrfir eve.py')
        # print(self.peek_dict(output_dict), '---egewfrfr eve.py')

        # Left-right consistency
        #if self.eye_side == 'binocular':
        if 'left_PoG_tobii' in input_dict and 'right_PoG_tobii' in input_dict:
            intermediate_dict['PoG_cm_initial_validity'] = (
                    input_dict['left_PoG_tobii_validity'] &
                    input_dict['right_PoG_tobii_validity']
            )
            # set the right_ so that it complies to base_loss_with_validity.py function where validity is retreived
            # from validity of the second argument (left, right), 'right_..._validity' in this case)
            intermediate_dict['right_PoG_cm_initial_validity'] = intermediate_dict['PoG_cm_initial_validity']

            # print(('right_PoG_cm_initial' in intermediate_dict.keys()), list(intermediate_dict.keys()), '---njgjeifjij eve.py')
            # print('line 688 mark', '---jfeijjfd eve.py')
            output_dict['loss_mse_lr_consistency'] = mse_loss(
                intermediate_dict['left_PoG_cm_initial'],
                'right_PoG_cm_initial', intermediate_dict,
            )
            output_dict['metric_euc_lr_consistency'] = euclidean_loss(
                intermediate_dict['left_PoG_cm_initial'],
                'right_PoG_cm_initial', intermediate_dict,
            )


        # Initial heatmap CE loss
        input_key = output_key = 'heatmap_initial'
        interm_key = ('heatmap_initial_unaugmented'
                      if self.training and config.refine_net_do_offset_augmentation
                      else 'heatmap_initial')
        if interm_key in intermediate_dict and input_key in input_dict:
            output_dict['loss_ce_' + output_key] = cross_entropy_loss(
                intermediate_dict[interm_key], input_key, input_dict,
            )

        # Refined heatmap MSE loss
        input_key = interm_key = 'heatmap_final'
        if interm_key in intermediate_dict and input_key in input_dict:
            output_dict['loss_ce_' + interm_key] = cross_entropy_loss(
                intermediate_dict[interm_key], input_key, input_dict,
            )
            output_dict['loss_mse_' + interm_key] = mse_loss(
                intermediate_dict[interm_key], input_key, input_dict,
            )

        # Metrics after applying kappa augmentation
        if config.refine_net_do_offset_augmentation:
            input_key = 'PoG_px_tobii'
            interm_key = 'PoG_px_initial_unaugmented'
            if interm_key in intermediate_dict and input_key in input_dict:
                output_dict['metric_euc_' + interm_key] = euclidean_loss(
                    intermediate_dict[interm_key], input_key, input_dict,
                )

            input_key = 'PoG_cm_tobii'
            interm_key = 'PoG_cm_initial_unaugmented'
            if interm_key in intermediate_dict and input_key in input_dict:
                output_dict['metric_euc_' + interm_key] = euclidean_loss(
                    intermediate_dict[interm_key], input_key, input_dict,
                )

            input_key = 'g'
            interm_key = 'g_initial_unaugmented'
            if interm_key in intermediate_dict and input_key in input_dict:
                output_dict['metric_ang_' + interm_key] = angular_loss(
                    intermediate_dict[interm_key], input_key, input_dict,
                )

        # Initial gaze
        input_key = 'PoG_px_tobii'
        interm_key = 'PoG_px_initial'
        if interm_key in intermediate_dict and input_key in input_dict:
            output_dict['loss_mse_' + interm_key] = mse_loss(
                intermediate_dict[interm_key], input_key, input_dict,
            )
            output_dict['metric_euc_' + interm_key] = euclidean_loss(
                intermediate_dict[interm_key], input_key, input_dict,
            )

        input_key = 'PoG_cm_tobii'
        interm_key = 'PoG_cm_initial'
        if interm_key in intermediate_dict and input_key in input_dict:
            output_dict['loss_mse_' + interm_key] = mse_loss(
                intermediate_dict[interm_key], input_key, input_dict,
            )
            output_dict['metric_euc_' + interm_key] = euclidean_loss(
                intermediate_dict[interm_key], input_key, input_dict,
            )

        input_key = 'g'
        interm_key = 'g_initial'
        if interm_key in intermediate_dict and input_key in input_dict:
            output_dict['metric_ang_' + interm_key] = angular_loss(
                intermediate_dict[interm_key], input_key, input_dict,
            )

        # Refine gaze
        input_key = 'PoG_px_tobii'
        interm_key = 'PoG_px_final'
        if interm_key in intermediate_dict and input_key in input_dict:
            output_dict['loss_mse_' + interm_key] = mse_loss(
                intermediate_dict[interm_key], input_key, input_dict,
            )
            output_dict['metric_euc_' + interm_key] = euclidean_loss(
                intermediate_dict[interm_key], input_key, input_dict,
            )

        input_key = 'PoG_cm_tobii'
        interm_key = 'PoG_cm_final'
        if interm_key in intermediate_dict and input_key in input_dict:
            output_dict['loss_mse_' + interm_key] = mse_loss(
                intermediate_dict[interm_key], input_key, input_dict,
            )
            output_dict['metric_euc_' + interm_key] = euclidean_loss(
                intermediate_dict[interm_key], input_key, input_dict,
            )

        input_key = 'g'
        interm_key = 'g_final'
        if interm_key in intermediate_dict and input_key in input_dict:
            output_dict['metric_ang_' + interm_key] = angular_loss(
                intermediate_dict[interm_key], input_key, input_dict,
            )

        if config.eye_validity_net_enabled:
            for side in ['left', 'right']:
                input_key = side + '_PoG_tobii_validity'
                interm_key = side + '_tracking_validity'
                if interm_key in intermediate_dict and input_key in input_dict:
                    output_dict['metric_bce_' + interm_key] = bec_loss_without_validity_check(
                        intermediate_dict[interm_key], input_dict[input_key].type(torch.FloatTensor).cuda(),
                    ) # loss_bce_left_tracking_validity
    def calculate_additional_labels(self, full_input_dict, current_epoch=None):
        sample_entry = next(iter(full_input_dict.values()))
        batch_size = sample_entry.shape[0]
        sequence_len = sample_entry.shape[1]

        # PoG in mm
        for side in ('left', 'right'):
            if (side + '_PoG_tobii') in full_input_dict:
                full_input_dict[side + '_PoG_cm_tobii'] = torch.mul(
                    full_input_dict[side + '_PoG_tobii'],
                    0.1 * full_input_dict['millimeters_per_pixel'],
                    ).detach()
                full_input_dict[side + '_PoG_cm_tobii_validity'] = \
                    full_input_dict[side + '_PoG_tobii_validity']

        # Fake kappa to be used during training
        # Mirror the yaw angle to handle different eyes
        if self.training and config.refine_net_do_offset_augmentation:

            # Curriculum learning on kappa
            # print(device, self.forward_kwargs['current_epoch'], current_epoch, '---nbwvffrf eve.py')
            assert(current_epoch is not None)
            assert(isinstance(current_epoch, float))
            kappa_std = config.refine_net_offset_augmentation_sigma
            kappa_std = np.radians(kappa_std)

            # Create systematic noise
            # This is consistent throughout a given sequence
            left_kappas = np.random.normal(size=(batch_size, 2), loc=0.0, scale=kappa_std)
            right_kappas = np.random.normal(size=(batch_size, 2), loc=0.0, scale=kappa_std)
            kappas = {
                'left': np.repeat(np.expand_dims(left_kappas, axis=1), sequence_len, axis=1),
                'right': np.repeat(np.expand_dims(right_kappas, axis=1), sequence_len, axis=1),
            }

            for side in ('left', 'right'):
                # Store kappa
                kappas_side = kappas[side]
                if not config.multi_gpu:
                    kappa_tensor = torch.tensor(kappas_side.astype(np.float32)).to(device)
                else:
                    kappa_tensor = torch.tensor(kappas_side.astype(np.float32)).cuda()
                full_input_dict[side + '_kappa_fake'] = kappa_tensor

        # 3D origin for L/R combined gaze
        if 'left_o' in full_input_dict:
            full_input_dict['o'] = torch.mean(torch.stack([
                full_input_dict['left_o'], full_input_dict['right_o'],
            ], axis=-1), axis=-1).detach()
            full_input_dict['o_validity'] = full_input_dict['left_o_validity']

        #print(list(full_input_dict.keys()), '---jefjiji, eve.py')
        if 'left_PoG_tobii' in full_input_dict:
            # Average of left/right PoG values
            full_input_dict['PoG_px_tobii'] = torch.mean(torch.stack([
                full_input_dict['left_PoG_tobii'],
                full_input_dict['right_PoG_tobii'],
            ], axis=-1), axis=-1).detach()
            full_input_dict['PoG_cm_tobii'] = torch.mean(torch.stack([
                full_input_dict['left_PoG_cm_tobii'],
                full_input_dict['right_PoG_cm_tobii'],
            ], axis=-1), axis=-1).detach()
            full_input_dict['PoG_px_tobii_validity'] = (
                    full_input_dict['left_PoG_tobii_validity'].bool() &
                    full_input_dict['right_PoG_tobii_validity'].bool()
            ).detach()
            full_input_dict['PoG_cm_tobii_validity'] = full_input_dict['PoG_px_tobii_validity']

            if config.refine_net_enabled:
                # Heatmaps (both initial and final)
                # NOTE: input is B x T x F
                full_input_dict['heatmap_initial'] = torch.stack([
                    batch_make_heatmaps(full_input_dict['PoG_px_tobii'][b, :],
                                        config.gaze_heatmap_sigma_initial)
                    * full_input_dict['PoG_px_tobii_validity'][b, :].float().view(-1, 1, 1, 1)
                    for b in range(batch_size)
                ], axis=0).detach()
                full_input_dict['heatmap_history'] = torch.stack([
                    batch_make_heatmaps(full_input_dict['PoG_px_tobii'][b, :],
                                        config.gaze_heatmap_sigma_history)
                    * full_input_dict['PoG_px_tobii_validity'][b, :].float().view(-1, 1, 1, 1)
                    for b in range(batch_size)
                ], axis=0).detach()
                full_input_dict['heatmap_final'] = torch.stack([
                    batch_make_heatmaps(full_input_dict['PoG_px_tobii'][b, :],
                                        config.gaze_heatmap_sigma_final)
                    * full_input_dict['PoG_px_tobii_validity'][b, :].float().view(-1, 1, 1, 1)
                    for b in range(batch_size)
                ], axis=0).detach()
                full_input_dict['heatmap_initial_validity'] = \
                    full_input_dict['PoG_px_tobii_validity']
                full_input_dict['heatmap_history_validity'] = \
                    full_input_dict['PoG_px_tobii_validity']
                full_input_dict['heatmap_final_validity'] = \
                    full_input_dict['PoG_px_tobii_validity']

        # 3D gaze direction for L/R combined gaze
        if 'PoG_cm_tobii' in full_input_dict:
            full_input_dict['g'] = torch.stack([
                calculate_combined_gaze_direction(
                    full_input_dict['o'][b, :],
                    10.0 * full_input_dict['PoG_cm_tobii'][b, :],
                    full_input_dict['left_R'][b, :],
                    full_input_dict['camera_transformation'][b, :],
                    )
                for b in range(batch_size)
            ], axis=0)
            full_input_dict['g_validity'] = full_input_dict['PoG_cm_tobii_validity']
    def from_g_to_PoG_history(self, full_input_dict, sub_input_dict, sub_output_dict,
                              input_suffix, output_suffix,
                              heatmap_history, gaze_heatmap_sigma, history_heatmap_sigma):

        # Handle case for GazeCapture and MPIIGaze
        if 'inv_camera_transformation' not in full_input_dict:
            return

        # Step 1a) Calculate PoG from given gaze
        # print('self.eye_side', self.eye_side, '---ejjijfje eve.py --from_g_to_PoG_history')
        #if self.eye_side == 'binocular':
        for side in ('left', 'right'):
            origin = (sub_output_dict[side + '_o']
                      if side + '_o' in sub_output_dict else sub_input_dict[side + '_o'])
            direction = sub_output_dict[side + '_g_' + input_suffix]
            rotation = (sub_output_dict[side + '_R']
                        if side + '_R' in sub_output_dict else sub_input_dict[side + '_R'])
            cc = [x.device.index for x in (origin, direction, rotation, sub_input_dict[side + '_R'])]
            # print('divice', torch.cuda.current_device(), cc, '---ejfjijiij eve.py')
            # if not all([xx == cc[0] for xx in cc]):
            #     print('divice', torch.cuda.current_device(), cc, '---vjjfjjje eve.py')
            PoG_mm, PoG_px = to_screen_coordinates(origin, direction, rotation, sub_input_dict)
            sub_output_dict[side + '_PoG_cm_' + output_suffix] = 0.1 * PoG_mm
            sub_output_dict[side + '_PoG_px_' + output_suffix] = PoG_px
        # print(self.peek_dict(sub_output_dict), '---bhjejfijie eve.py')
        # print(list(sub_output_dict.keys()), '---ngwefrdsf eve.py')

        # Step 1b) Calculate average PoG
        sub_output_dict['PoG_px_' + output_suffix] = torch.mean(torch.stack([
            sub_output_dict['left_PoG_px_' + output_suffix],
            sub_output_dict['right_PoG_px_' + output_suffix],
        ], axis=-1), axis=-1)
        sub_output_dict['PoG_cm_' + output_suffix] = torch.mean(torch.stack([
            sub_output_dict['left_PoG_cm_' + output_suffix],
            sub_output_dict['right_PoG_cm_' + output_suffix],
        ], axis=-1), axis=-1)

        ############################################################################################
        # 1. store original eye_net output to build memory
        sub_output_dict['PoG_px_' + output_suffix + '_for_memory'] = sub_output_dict['PoG_px_' + output_suffix][:]
        sub_output_dict['PoG_cm_' + output_suffix + '_for_memory'] = sub_output_dict['PoG_cm_' + output_suffix][:]
        #pxs, cms = sub_output_dict['PoG_px_' + output_suffix], sub_output_dict['PoG_cm_' + output_suffix]
        cms = sub_output_dict['PoG_cm_' + output_suffix]
        rk = lambda k: sub_output_dict[k][:].detach().cpu().numpy()[0]
        cm_np_l, cm_np_r = rk('left' + '_PoG_cm_' + output_suffix), rk('right' + '_PoG_cm_' + output_suffix)

        if config.central_calibration_enabled:
            # centers are calculated by averaging the ground truth PoG_px, PoG_cm in all samples in training set (in Metric 00. eval inference output + retrieve screen center)
            #print(sub_output_dict['PoG_px_' + output_suffix], '---jeiiiii evec.py')
            # 2. reteive memory visual center
            # memory = self.retrieve_memory_visual_center()
            #st()
            assert len(full_input_dict['participant']) == 1
            participant, cam_pos, video_id = full_input_dict['participant'][0], full_input_dict['camera'][0], full_input_dict['subfolder'][0]

            self.rearrange_memory_if_needed(participant, cam_pos, video_id)
            cms_center_calibrated, cms_st_refined, is_valid = \
                self.center_calibration_and_st_refine(participant, cam_pos, video_id, cms, cm_np_l, cm_np_r)

            if self.plot_this_sample:
                self.plotting_materials_one_sample['PoG_cm_initial'] = cms[0].detach().cpu().numpy()
                self.plotting_materials_one_sample['PoG_cm_after_sc'] = cms_center_calibrated[0].detach().cpu().numpy()
                self.plotting_materials_one_sample['PoG_cm_after_sc_st'] = cms_st_refined[0].detach().cpu().numpy()
                #st()

            #sub_output_dict['PoG_cm_' + output_suffix] = cms_center_calibrated
            sub_output_dict['PoG_cm_' + output_suffix] = cms_st_refined
            # calculate PoG_px from PoG_cm
            sub_output_dict['PoG_px_' + output_suffix] = \
                self.PoG_cm_to_px_on_screen(sub_output_dict['PoG_cm_' + output_suffix], sub_input_dict)
            # sub_output_dict['predicted_tracking_validity_' + output_suffix] = torch.from_numpy(np.array([is_valid,]))

        ######################################################################################################

        # if datetime.datetime.now().strftime('%Y%m%d%H%M%S') > '20210220192900':
        #     torch.save(sub_output_dict['PoG_px_' + output_suffix], '/samba/room/storage/eve_dataset/temp_pog')
        #     raise


        sub_output_dict['PoG_mm_' + output_suffix] = \
            10.0 * sub_output_dict['PoG_cm_' + output_suffix]

        # Step 1c) Calculate the combined gaze (L/R)
        sub_output_dict['g_' + output_suffix] = \
            calculate_combined_gaze_direction(
                sub_input_dict['o'],
                sub_output_dict['PoG_mm_' + output_suffix],
                sub_input_dict['left_R'],  # by definition, 'left_R' == 'right_R'
                sub_input_dict['camera_transformation'],
            )

        if config.refine_net_enabled:
            # Step 2) Create heatmaps from PoG estimates
            sub_output_dict['heatmap_' + output_suffix] = \
                batch_make_heatmaps(sub_output_dict['PoG_px_' + output_suffix], gaze_heatmap_sigma)

            if heatmap_history is not None:
                # Step 3) Create gaze history maps
                heatmap_history.append(
                    batch_make_heatmaps(sub_output_dict['PoG_px_' + output_suffix],
                                        history_heatmap_sigma)
                )
                if 'PoG_px_tobii' in full_input_dict:
                    gaze_history_maps = batch_make_gaze_history_maps(
                        full_input_dict['timestamps'], heatmap_history,
                        full_input_dict['PoG_px_tobii_validity'],
                    )
                    sub_output_dict['history_' + output_suffix] = gaze_history_maps
    def PoG_cm_to_px_on_screen(self, PoG_cm, input_dict):
        PoG_mm = PoG_cm * 10.0
        # Convert back from mm to pixels
        ppm_w = input_dict['pixels_per_millimeter'][:, 0]
        ppm_h = input_dict['pixels_per_millimeter'][:, 1]
        PoG_px = torch.stack([
            torch.clamp(PoG_mm[:, 0] * ppm_w,
                        0.0, float(config.actual_screen_size[0])),
            torch.clamp(PoG_mm[:, 1] * ppm_h,
                        0.0, float(config.actual_screen_size[1]))
        ], axis=-1)
        return PoG_px


    ## methods for st_refine fixed length history in forward()
    # level 1 methods
    def rearrange_memory_if_needed(self, subject, cam_pos, video_id):
        if self.fixed_history_len is None: #
            return
        else:
            #print(subject, cam_pos, video_id, '--efjijoi evec')
            expected_memory_version = '##'.join([subject, cam_pos, video_id, 'initiated'])
            #st()
            if self.running_memory_version != expected_memory_version:
                self.initiate_fixed_length_memory_for_video(subject, cam_pos, video_id)
                self.running_memory_version = expected_memory_version
    # level 2 methods for rearrange_memory_if_needed
    def initiate_fixed_length_memory_for_video(self, subject, cam_pos, video_id):
        #st()
        history_pi = self.fixed_length_history[subject][cam_pos][video_id]['pi']
        history_bi = self.fixed_length_history[subject][cam_pos][video_id]['bi']
        self.write_empty_memories()
        N = len(history_pi)
        with torch.no_grad():
            for idx in range(N): # run through history pi in order to accmulate memory based on pi
                cm_np = history_pi[idx]
                pre_validity = history_bi[idx]
                cms = torch.from_numpy(cm_np.reshape(-1, 2)).float().cuda()
                cms_center_calibrated, cms_st_refined, is_valid = self.center_calibration_and_st_refine(subject, cam_pos, video_id, cms, mute_st=True)
        #print('memory initiated for', subject, cam_pos, video_id, '---ekjiij do_st_refine.py')

    # methods for self-calibration and st_refinement
    # level 0 method
    def center_calibration_and_st_refine(self, participant, cam_pos, video_id, cms, cm_np_l=None, cm_np_r=None, pre_validity=None, mute_st=False):
        memory_pi, memory_bi, memory_pi_star, memory_pi_m_std, center = self.retrieve_memory_for_participant(participant, cam_pos)

        memory_visual_center = center # updated after every 1% new data added
        #print(center, '---jfjoijej evec.py')

        cms_center_calibrated = None
        cms_st_refined = None
        if memory_visual_center is not None:
            # 1. center calibration
            # sub_output_dict['PoG_px_' + output_suffix] = \
            #     self.calibrate_to_center(pxs, subject_memory_center=memory['PoG_px'][0], avg_center=np.array([909.6951, 529.9885]))
            cms_center_calibrated = self.calibrate_to_center(cms, subject_memory_center=memory_visual_center,
                                                             avg_center=np.array([SCREEN_CENTER_CM_X, SCREEN_CENTER_CM_Y]))

            # 2. do spacial transform before clamping px
            if config.st_transform_enabled:
                if not mute_st:
                    #if len(memory_pi_star) >= 3000: # st should fail when gaze path shape is not complete
                    if len(memory_bi) > 0 and np.sum(memory_bi) >= self.online_refinement_starts_from: # at least 0s memory is required for spatial transform
                        PoG_pxs = self.PoG_cm_to_px_without_clampping(cms_center_calibrated)
                        assert PoG_pxs.shape[0] == 1
                        PoG_cm_history = [pi_star for pi_star, bi in zip(memory_pi_star, memory_bi) if bi > 0.5] # select valid pi_stars for history heatmap
                        PoG_px_history =self.PoG_cm_to_px_without_clampping(torch.from_numpy(np.array(PoG_cm_history)))
                        PoG_px_refined, is_refined, input_combmap, output_combmap = self.st_refine_net.refine(PoG_pxs[0], PoG_px_history, return_full=True)
                        PoG_pxs_refined = PoG_px_refined.reshape(-1, 2)
                        cms_st_refined = self.PoG_px_to_cm(PoG_pxs_refined)
                        #print(cms_st_refined, '---ejjiisdkj evec.py')
                        #sub_output_dict['PoG_cm_' + output_suffix] = PoG_cm_refined
                        #print('refined from', PoG_pxs[0], 'to', PoG_px_refined[0], 'is_refined', is_refined, '---jjfjiijjd')
                        if self.plot_this_sample:
                            self.plotting_materials_one_sample['PoG_cm_history'] = np.array(PoG_cm_history)
                            self.plotting_materials_one_sample['PoG_px_history'] = np.array(PoG_px_history)
                            self.plotting_materials_one_sample['ST_input_combmap'] = input_combmap
                            if output_combmap is not None:
                                self.plotting_materials_one_sample['ST_output_combmap'] = output_combmap[0]
                            else:
                                self.plotting_materials_one_sample['ST_output_combmap'] = None

        if (cms_center_calibrated is None) or not torch.all(torch.isfinite(cms_center_calibrated)):
            cms_center_calibrated = cms # when center calibration has not started to work, pass pi_intial to memory_pi_star
        if (cms_st_refined is None)  or not torch.all(torch.isfinite(cms_st_refined)):
            cms_st_refined = cms_center_calibrated

        # 4. update memory for each frame passed
        #pxs_np, cms_np = pxs.detach().cpu().numpy(), cms.detach().cpu().numpy()
        cm_np = cms.detach().cpu().numpy()[0] # the cms is a torch array of shape (1, 2), i.e. [[cm_x, cm_y]] with one datum only
        center_calibrated_cm_np = cms_center_calibrated.detach().cpu().numpy()[0] # when center calibration has not started to work, pass pi_intial to memory_pi_star
        if pre_validity is None:
            is_valid = self.check_eye_tracking_validity(cm_np, cm_np_l, cm_np_r, memory_pi_m_std)
        else:
            is_valid = pre_validity
        self.update_subject_memory(participant, cam_pos, video_id, is_valid, cm_np, center_calibrated_cm_np)
        #print(cms_center_calibrated, cms_st_refined, '---fjeofijrji evec.py')
        return cms_center_calibrated, cms_st_refined, is_valid
    def save_subject_memory(self, storage_file_path):
        ta = (self.memory_pi, self.memory_bi, self.memory_pi_star, self.memory_pi_mean_std, self.memory_pi_mean_after_outlier_removal, self.memory_video_id, self.memory_last_update_at)
        pickle.dump(ta, open(storage_file_path, 'wb'))
    def load_subject_memory(self, storage_file_path):
        ta = pickle.load(open(storage_file_path, 'rb'))
        self.memory_pi, self.memory_bi, self.memory_pi_star, self.memory_pi_mean_std, self.memory_pi_mean_after_outlier_removal, self.memory_video_id, self.memory_last_update_at = ta

    # level 1 methods for center_calibration_and_st_refine()
    def retrieve_memory_for_participant(self, participant, cam_pos):
        if participant not in self.memory_pi.keys():
            self.memory_pi[participant] = {}
            self.memory_bi[participant] = {}
            self.memory_pi_star[participant] = {}
            self.memory_pi_mean_std[participant] = {}
            self.memory_pi_mean_after_outlier_removal[participant] = {}
            self.memory_video_id[participant] = {}
            self.memory_last_update_at[participant] = {}

        if cam_pos not in self.memory_pi[participant].keys():
            #st()
            self.memory_pi[participant][cam_pos] = []
            self.memory_bi[participant][cam_pos] = []
            self.memory_pi_star[participant][cam_pos] = []
            self.memory_pi_mean_after_outlier_removal[participant][cam_pos] = None
            self.memory_pi_mean_std[participant][cam_pos] = ((None, None), (None, None)) # ((mean_x, std_x), (mean_y, std_y))
            self.memory_video_id[participant][cam_pos] = []
            self.memory_last_update_at[participant][cam_pos] = -1
        #st()
        # print(self.memory_pi[participant][cam_pos], self.memory_bi[participant][cam_pos], \
        #       self.memory_pi_star[participant][cam_pos], self.memory_pi_mean_std[participant][cam_pos], '---ejkfjjeoi evec.py')
        #st()
        #print(self.memory_pi_mean_after_outlier_removal, '---kfeijijij evec.py')
        return self.memory_pi[participant][cam_pos], self.memory_bi[participant][cam_pos], \
               self.memory_pi_star[participant][cam_pos], self.memory_pi_mean_std[participant][cam_pos], \
               self.memory_pi_mean_after_outlier_removal[participant][cam_pos]
    def calibrate_to_center(self, pxs, subject_memory_center, avg_center):
        #print(pxs, subject_memory_center, avg_center, '---jejiffjjj evec.py')
        error = torch.from_numpy(subject_memory_center - avg_center).float().cuda()
        pxs = pxs - error
        #print(pxs, '---bvvhfuej evec.py')
        return pxs
    def check_eye_tracking_validity(self, cm_np, cm_np_l, cm_np_r, memory_pi_m_std):
        xx, yy = tuple(cm_np)
        (mean_x, std_x), (mean_y, std_y) = memory_pi_m_std
        if (cm_np_l is not None) and (np.isfinite(cm_np_l[0])):
            with_in_screen = self.check_if_PoG_cm_within_screen(cm_np_l) and self.check_if_PoG_cm_within_screen(cm_np_r)
        else:
            with_in_screen = self.check_if_PoG_cm_within_screen(cm_np)

        # calculate if cm_np is not an outlier of existing data points in memory
        if mean_x is not None:
            xx, yy = tuple(cm_np)
            with_in_center = (abs(xx - mean_x) < std_x * self.center_drag) and (abs(yy - mean_y) < std_y * self.center_drag)
        else:
            with_in_center = False

        validity = with_in_screen or with_in_center
        return validity
    def update_subject_memory(self, participant, cam_pos, video_id, is_valid, cm_np, center_calibrated_cm_np):
        # 1. store data points
        self.memory_pi[participant][cam_pos].append(cm_np)
        self.memory_bi[participant][cam_pos].append(int(is_valid))
        self.memory_pi_star[participant][cam_pos].append(center_calibrated_cm_np)
        self.memory_video_id[participant][cam_pos].append(video_id)
        # 2. update mean and std for memory_pi
        mii = len(self.memory_bi[participant][cam_pos])
        mii_pre = self.memory_last_update_at[participant][cam_pos]
        if (mii_pre == -1) or ((mii - mii_pre)/(mii_pre+1) >= 0.01): # skip, if number of uncalculated new memory is more than 1% of the calculated ones, update
            #st()
            memory_pi_valid = np.array([pi for pi, bi in zip(self.memory_pi[participant][cam_pos], self.memory_bi[participant][cam_pos]) if bi > 0.5])
            if len(memory_pi_valid) > 0:
                center, (m_x, std_x, m_y, std_y) = self.calculate_memory_center(memory_pi_valid, num_std=self.memory_center_calculation_out_outlier_thresh)
                if (center is not None) and (np.isfinite(center[0])):
                    self.memory_pi_mean_after_outlier_removal[participant][cam_pos] = center
                    self.memory_pi_mean_std[participant][cam_pos] = ((m_x, std_x), (m_y, std_y))
                    self.memory_last_update_at[participant][cam_pos] = mii
                    self.count_memory_update += 1
                    if self.count_memory_update % 1000 == 0:
                        print(participant, cam_pos, 'memory updated at', mii, 'update percent', '{:.3%}'.format((mii - mii_pre)/(mii_pre+1)) if mii_pre > -1 else 'not applicable', '---kekjfj evec.py')

    def PoG_px_to_cm(self, PoG_px):
        PoG_cm = torch.stack([PoG_px[:, 0]*CM_PER_PIXEL_X, PoG_px[:, 1]*CM_PER_PIXEL_Y], axis=-1)
        return PoG_cm
    def PoG_cm_to_px_without_clampping(self, PoG_cm):
        PoG_px = torch.stack([PoG_cm[:, 0]/CM_PER_PIXEL_X, PoG_cm[:, 1]/CM_PER_PIXEL_Y], axis=-1)
        return PoG_px
    # level 2 method for check_eye_tracking_validity()
    def check_if_PoG_cm_within_screen(self, cm):
        # PIXEL_PER_DEGREE = 42.082546   # calculated by averaging the 'metric_euc_PoG_px_intial'/'metric_ang_g_initial' for each clip in eye_net results on validation data
        # MARGIN_CM_ONE_DEGREE = 1.212 # calculated by PIXEL_PER_ANGLE * 1.0 * m2p_x or _y / 10.0
        # SCREEN_W_CM, SCREEN_H_CM = 55.3000, 31.1000
        # print(cm, '---efiiiijd')
        x, y = tuple(cm)
        # enlarged valid area (fixation outside screen can be valid for eye tracker as long as the pupil is not missing in camera)
        validity_x = (0 - MARGIN_CM_ONE_DEGREE) < x < (SCREEN_W_CM + MARGIN_CM_ONE_DEGREE)
        validity_y = (0 - MARGIN_CM_ONE_DEGREE) < y < (SCREEN_H_CM + MARGIN_CM_ONE_DEGREE)
        validity = validity_x and validity_y
        #print(x, y, validity, '---ejjjiidi evec.py')
        #if not validity: print('invalid cm predicted', cm, '---jjiejoij evec.py')
        return validity
    # level 2 method for update_subject_memory()
    def calculate_memory_center(self, xys, num_std):
        xs, ys = xys[:, 0], xys[:, 1]
        xs_good, ys_good, (m1, std1, m2, std2) = self.remove_outliers_for_two_arrs(xs, ys, num_std)
        center = np.array([np.mean(xs_good), np.mean(ys_good)])
        #print(xys, xs_good, ys_good, center, (m1, std1, m2, std2), '---ejfijoij')
        #print(self.memory_pi, self.memory_bi, '---ejfiojoi')
        return center, (m1, std1, m2, std2)
    # level 3 method for calculate_memory_center()
    def remove_outliers_for_two_arrs(self, arr1, arr2, num_std=3):
        marks1, m1, std1 = self.generate_outlier_mark(arr1, num_std)
        marks2, m2, std2 = self.generate_outlier_mark(arr2, num_std)
        marks = [a and b for a, b in zip(marks1, marks2)]
        return arr1[marks], arr2[marks], (m1, std1, m2, std2)
    # level 4 method for remove_outliers_for_two_arrs()
    def generate_outlier_mark(self, arr, num_std):
        m, std = np.mean(arr), np.std(arr)
        marks_keep = abs(arr - m) < num_std * std
        return marks_keep, m, std

    # obsolete
    def retrieve_memory_visual_center(self, full_input_dict):
        memory_path = self.get_memory_visual_center_path(full_input_dict)
        if os.path.isfile(memory_path):
            memory = pickle.load(open(memory_path, 'rb'))
        else:
            #memory = {'PoG_cm': None, 'PoG_px': None}
            memory = {'PoG_cm_x_list': [], 'PoG_cm_y_list': [], 'PoG_cm_memory_after_center_calibration':[],
                      'PoG_cm':None, 'PoG_cm_m_std':None,
                      'num_valid':0, 'num_invalid':0, 'last_update_point_num_valid':0, 'drag_in_count':0,
                      'PoG_cm_l_curr': None, 'PoG_cm_l_pre1': None, 'pupil_l_curr': None, 'pupil_l_pre1': None,
                      'PoG_cm_r_curr': None, 'PoG_cm_r_pre1': None, 'pupil_r_curr': None, 'pupil_r_pre1': None }
        return memory
    def get_memory_visual_center_path(self, full_input_dict):
        if config.output_path != '':
            path_tokens = config.output_path.split('/') # when running inference with src/inference.py
            #print(path_tokens, '---fjjijjdf evec.py')
            subject = path_tokens[-3]
            camera_pos = path_tokens[-1][:-5]
            memory_path = '/'.join(path_tokens[:-2]) + '/' + 'memory_visual_center_' + camera_pos
            # print(memory_path, '---ejiji evec.py')
        else:
            #print(list(full_input_dict.keys()), '---ejfjiji evec.py')
            # '/samba/room/storage/Bji//network_outputs/inference_day22_inference_eye_net_test/val01/memory_visual_center_basler'
            # /samba/room/storage/Bji/ test07 step096_image_MIT-i1064228254 webcam_l
            participant = full_input_dict['participant'][0]
            subfolder = full_input_dict['subfolder'][0]
            camera = full_input_dict['camera'][0]
            memory_path = config.output_dir + 'network_outputs/eval_codalab_' + config.config_id + '/' + \
                          participant + '/' + 'memory_visual_center_' + camera
        return memory_path
    def calculate_memory_visual_center(self, memory_pi, memory_bi):
        assert len(memory_bi) == len(memory_pi)
        if len(memory_pi) == 0:
            return None
        else:
            he = np.multiply(np.array(memory_pi), np.array(memory_bi).reshape(-1,1))
            avg = np.sum(he, axis=0)/np.sum(memory_bi)
            return avg
    def check_tracking_validity(self, sub_output_dict, output_suffix, memory_visual_center):

        # retrive curr data
        md = self.get_binocular_PoG_history(sub_output_dict, output_suffix, memory_visual_center)

        # calculate validity
        with_in_screen = self.check_if_PoG_cm_within_screen(md['cm_l']) and self.check_if_PoG_cm_within_screen(md['cm_r'])
        # pupil_on = (md['pupil_l'] > 1.0) and (md['pupil_r'] > 1.0) # predicted pupil size below 1.0 coinsides with invalid tobii data

        # calculate exempt
        if memory_visual_center['PoG_cm_m_std'] is not None:
            m1, std1, m2, std2 = memory_visual_center['PoG_cm_m_std']
            xx, yy = (md['cm_l'][0] + md['cm_r'][0])/2, (md['cm_l'][1] + md['cm_r'][1])/2,
            with_in_center = (abs(xx - m1) < std1 * self.center_drag) and (abs(yy - m2) < std2 * self.center_drag)
        else:
            with_in_center = False

        # when memory is large enough, should include samples that are close to meomory center, even if they are outof screen
        if memory_visual_center['num_valid'] < 1000:
            validity = with_in_screen
        else:
            validity = with_in_screen or with_in_center
            if with_in_center and not with_in_screen:
                print('out of screen but close to memory center, stay in memory', 'l', md['cm_l'], 'r', md['cm_r'], '---jifjij evec')
                memory_visual_center['drag_in_count'] += 1

        # if md['cm_l_pre2'] is not None: # (stability check disabled)
        #     is_near = lambda a, b: np.sqrt(np.sum((a - b)**2)) < CM_PER_DEGREE * self.memory_in_stable_thresh # we mark PoG pairs less than 3 degrees apart as near
        #     # if a PoG is near to its two predecessors, we mark it as stable
        #     is_stable = is_near(md['cm_l'], md['cm_l_pre1']) and is_near(md['cm_l'], md['cm_l_pre2']) and \
        #                 is_near(md['cm_r'], md['cm_r_pre1']) and is_near(md['cm_r'], md['cm_r_pre2'])
        #     validity = validity and is_stable

        return validity, md
    def get_memory_visual_center(self, participant, cam_pos):
        center =  self.memory_pi_mean_after_outlier_removal[participant][cam_pos]
        return center
    def get_binocular_PoG_history(self, sub_output_dict, output_suffix, memory_visual_center):
        rk = lambda k: sub_output_dict[k][:].detach().cpu().numpy()[0]
        dd = {}
        dd['cm_l'] = rk('left' + '_PoG_cm_' + output_suffix)
        dd['cm_r'] = rk('right' + '_PoG_cm_' + output_suffix)
        dd['pupil_l'] = rk('left_pupil_size')
        dd['pupil_r'] = rk('right_pupil_size')

        # retrive previous data
        dd['cm_l_pre1'] = memory_visual_center['PoG_cm_l_curr']
        dd['cm_r_pre1'] = memory_visual_center['PoG_cm_r_curr']
        dd['cm_l_pre2'] = memory_visual_center['PoG_cm_l_pre1']
        dd['cm_r_pre2'] = memory_visual_center['PoG_cm_r_pre1']
        dd['pupil_l_pre1'] = memory_visual_center['pupil_l_curr']
        dd['pupil_r_pre1'] = memory_visual_center['pupil_r_curr']
        dd['pupil_l_pre2'] = memory_visual_center['pupil_l_pre1']
        dd['pupil_r_pre2'] = memory_visual_center['pupil_r_pre1']
        return dd
    def clear_memory_visual_center(self, output_path):
        pass
    def combine_avg(self, avg1, cnt1, avg2, cnt2):
        cnt = cnt1 + cnt2
        assert cnt > 0
        avg = (avg1 * cnt1 + avg2 * cnt2)/cnt
        return avg, cnt
    def generate_eye_list(self):
        if self.eye_side in ['left', 'right']:
            eye_list = [self.eye_side,]
        elif self.eye_side == 'binocular':
            eye_list = ['left', 'right']
        else:
            raise
        return eye_list

    # miscellanies
    def peek_dict(self, dd, suffix=''):
        reports = []
        for k, v in dd.items():
            if type(v) == list:
                rr = ((k, len(v)))
            elif type(v) == dict:
                rr = ((k, self.peek_dict(v)))
            else:
                rr = ((k, v.size()))
            reports.append(rr)
        # print(reports, '---fuefjj', suffix)
        return reports

# utiliits
def createDir(dstDir):
    if not os.path.exists(dstDir):
        os.makedirs(dstDir)
def check_dict(d, indent=0):
    for key, value in list(d.items())[:1]:
        print('\t' * indent + str(key) + ' --- ' + str(len(d)))
        if isinstance(value, dict):
            check_dict(value, indent+1)
        else:
            #print('\t' * (indent+1) + str(value))
            print('\t' * (indent) + 'ENDs with keys: ' + str(list(d.keys())))
