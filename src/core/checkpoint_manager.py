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
import glob
import logging
import os
import shutil
import pickle

import torch

from core import DefaultConfig

config = DefaultConfig()
logger = logging.getLogger(__name__)

# device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
curr_device_index = torch.cuda.current_device()
#device = torch.device('cuda:'+str(curr_device_index)  if torch.cuda.is_available() else "cpu")
device = torch.device(("cuda:0" if config.use_one_gpu is None else 'cuda:'+str(config.use_one_gpu) )  if torch.cuda.is_available() else "cpu")
device = torch.device('cuda:0')
print(device, '---tegrferg core/checkpoint_manager.py')

class CheckpointManager(object):
    __model = None
    __optimizers = None
    __suffix = '.pt'

    def __init__(self, model, optimizers):
        self.__model = model
        self.__optimizers = optimizers
        self.history_stats = {} # initialise history_stat as empty dict

    def __save(self, ofdir):
        assert not os.path.isdir(ofdir)
        if hasattr(self.__model, 'module'):  # case where nn.DataParallel was used
            state_dict = self.__model.module.state_dict()
        else:
            state_dict = self.__model.state_dict()
        os.makedirs(ofdir)

        # Determine prefices
        prefices = set()
        for k in state_dict.keys():
            words = k.split('.')
            prefices.add(words[0])

        # Save each prefix into own file
        for prefix in prefices:
            sub_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith(prefix + '.'):
                    sub_state_dict[k] = v
            torch.save(sub_state_dict, '%s/%s%s' % (ofdir, prefix, self.__suffix))

        # Save each optimizer's state
        for i, optimizer in enumerate(self.__optimizers):
            output_path = '%s/optimizer_%d%s' % (ofdir, i, self.__suffix)
            torch.save(optimizer.state_dict(), output_path)

        # # Save history stats and clear slot
        # history_stats_dir = '%s/history_stats_%s' % (ofdir, self.__suffix)
        # torch.save(self.history_stats, history_stats_dir)
        # print('history_stats stored in', history_stats_dir, '---woejfuhfu')
        # self.history_stats = {}

        logger.info('> Saved parameters to: %s' % ofdir)
        print('check point saved at step', ofdir, '---urghiufj')

    def __load(self, ifdir):
        assert os.path.isdir(ifdir)
        full_state_dict = {}

        # Gather state_dicts from directory
        ifpaths = [
            p for p in glob.glob(ifdir + '/*' + self.__suffix)
            if os.path.isfile(p) and not os.path.basename(p).startswith('optimizer_')
        ]
        for ifpath in ifpaths:
            sub_state_dict = torch.load(ifpath, map_location=device)
            for k, v in sub_state_dict.items():
                full_state_dict[k] = v
            logger.info('> Loaded model parameters from: %s' % ifpath)

        # Do the actual loading
        # print(full_state_dict['fc_to_gaze.0.weight'], full_state_dict['fc_to_gaze.0.weight'].size(), '---bwfrffd checkpoint_manager.py')
        self.__model.load_state_dict(full_state_dict, strict=False) # allow other dictionary and keys to be stored in checkpoint
        # print(self.__model.state_dict()['fc_to_gaze.0.weight'],
        #       self.__model.state_dict()['fc_to_gaze.0.weight'].size(), '---nbvdwfrfrfe checkpoint_manager.py')
        #raise


        # Load each optimizer's state
        optimizer_checkpoint_paths = [
            p for p in glob.glob(ifdir + '/optimizer_*' + self.__suffix)
            if os.path.isfile(p)
        ]
        for checkpoint_path in optimizer_checkpoint_paths:
            optimizer_index = int(os.path.basename(checkpoint_path).split('.')[0].split('_')[-1])
            if optimizer_index < len(self.__optimizers):
                self.__optimizers[optimizer_index].load_state_dict(
                    torch.load(checkpoint_path, map_location=device)
                )
                logger.info('> Loaded optimizer parameters from: %s' % checkpoint_path)

        # load history stats #TODO: no need to load history to model or checkpoint manager
        # self.history_stats = torch.load('%s/history_stats_%s' % (ifdir, self.__suffix))

        step = int(os.path.split(ifdir)[-1][:-3])
        return step

    def __output_dir(self):
        return os.path.relpath(os.path.join(
            self.__model.output_dir,
            'checkpoints',
        ))

    def __output_fpath(self, current_step, is_final_step=False):
        return os.path.relpath(os.path.join(
            self.__output_dir(),
            ('%07d' % current_step if not is_final_step else 'final_checkpoint') + self.__suffix,
        ))

    def save_at_step(self, current_step, is_final_step=False):
        self.__save(self.__output_fpath(current_step, is_final_step))
        self.__only_keep_n_checkpoints()

    def save_at_step_training_results(self, current_step, training_results):
        ofdir = self.__output_fpath(current_step)
        # ofdir example: '..\outputs\EVE\201202_102954.20b18b\checkpoints\0000005.pt'
        step_str = ofdir[-10:-3]
        checkpoints_dir = ofdir[:-11]
        training_results_dir = checkpoints_dir + '/training_results'
        if not os.path.isdir(training_results_dir): os.makedirs(training_results_dir)
        step_train_results_dir = training_results_dir + '/training_results_' + step_str + '' + self.__suffix
        torch.save(training_results, step_train_results_dir)
        # print('traing result list saved at step', current_step, 'in', step_train_results_dir, '---rgwfwefd')

    def save_at_step_testing_results(self, current_step, testing_results):
        ofdir = self.__output_fpath(current_step)
        # ofdir example: '..\outputs\EVE\201202_102954.20b18b\checkpoints\0000005.pt'
        step_str = ofdir[-10:-3]
        checkpoints_dir = ofdir[:-11]
        test_results_dir = checkpoints_dir + '/test_results'
        if not os.path.isdir(test_results_dir): os.makedirs(test_results_dir)
        step_test_results_dir = test_results_dir + '/test_results_' + step_str + '' + self.__suffix
        torch.save(testing_results, step_test_results_dir)
        # print('test results saved at step', current_step, 'in', step_test_results_dir, '---aefwedffd')

    def __get_available_checkpoints(self):
        output_dir = self.__output_dir()
        return sorted([
            (int(os.path.split(fn)[-1].split('.')[0]), fn)
            for fn in glob.glob(os.path.join(output_dir, '*' + self.__suffix))
            if fn.endswith(self.__suffix) and os.path.isdir(fn)
        ])

    def __only_keep_n_checkpoints(self):
        available = self.__get_available_checkpoints()
        if len(available) > config.checkpoints_keep_n:
            for step, fpath in available[:-config.checkpoints_keep_n]:
                shutil.rmtree(fpath)
                logger.info('> Removing parameters folder at: %s' % fpath)

    def load_last_checkpoint(self):
        return self.__load_last_checkpoint()

    def __load_last_checkpoint(self):
        available = self.__get_available_checkpoints()
        if len(available) > 0:
            return self.__load(available[-1][1])
        else:
            return 0
