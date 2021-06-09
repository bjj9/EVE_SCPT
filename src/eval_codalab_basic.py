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
import gzip
import logging
import os
import pickle
import time
import numpy as np
import torch

from core.config_default import DefaultConfig
import core.eval_codalab as eval_codalab
from models.evec import EVEC

# input paths and output dirs
WORKING_DIR_ROOT = '..' # set to '' when the current dir is './EVE_SCPT'
json_file_mark = 'inference_eye_net_10Hz_without_pupil_valid_center_calibrated_st' # error 1.95
input_dir = WORKING_DIR_ROOT + '/inputs/datasets/eve_dataset'
output_dir = WORKING_DIR_ROOT + '/outputs/eval_results'
memory_dir = WORKING_DIR_ROOT + '/memories'
memory_path = memory_dir + '/EVE_D_e_refine_stpt_np_vm999_cco1_cd3_test195_subject_eye_net_memories'

def eval_codalab_basic(output_suffix, skip_first_round_if_memory_is_ready=False):
    # Default singleton config object
    config = DefaultConfig()

    # Set device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Check memory: If there is no memory stored, this is the online evaluation or first round; Memory will be stored after online evaluation.
    # Offline evaluation can be performed only when memory has been stored.
    memory_is_ready = os.path.isfile(memory_path) # if there is no first round memory
    first_round = not memory_is_ready
    print('is first round (online) or not (offline)', first_round, '---jjjeiiif')
    if skip_first_round_if_memory_is_ready and memory_is_ready:
        print('Memory is ready. Skip first round and do offline evaluation directly.', '---jdkjdffjk')
        return
    # do change the config_id of this json for setting up a new storage place
    default_model_config = WORKING_DIR_ROOT + '/src/configs/' + json_file_mark + '.json'
    eval_codalab.script_init_common(default_model_config)

    # Initialize dataset and dataloader
    dataset, dataloader = eval_codalab.init_dataset()

    # # Define and set up model
    if first_round:
        # # ## online model
        online_refinement_starts_from = 2000 # tested by fixed randome history length # 18% of average data length for one subject one camera
        model = EVEC(output_predictions=True, online_refinement_starts_from=online_refinement_starts_from).to(device)
    else:
        # fixed memory length model (load from existing memory)
        input_memory_path = memory_path
        model = EVEC(output_predictions=True,
                     fixed_history_len='full', input_memory_path=input_memory_path).to(device)

    if config.resume_from != '':
        model = eval_codalab.model_setup(model)

    # Do eval_codalab
    processed_so_far = set()
    outputs_to_write = {}
    for step, inputs, outputs in eval_codalab.iterator(model, dataloader):
        batch_size = next(iter(outputs.values())).shape[0]

        for i in range(batch_size):
            participant = inputs['participant'][i]
            subfolder = inputs['subfolder'][i]
            camera = inputs['camera'][i]

            # Ensure that the sub-dicts exist.
            if participant not in outputs_to_write:
                outputs_to_write[participant] = {}
            if subfolder not in outputs_to_write[participant]:
                outputs_to_write[participant][subfolder] = {}

            # Store back to output structure
            keys_to_store = [
                'timestamps',
                'left_pupil_size',
                'right_pupil_size',
                'PoG_px_initial',
                'PoG_px_final',
                #'predicted_tracking_validity_final',
            ]
            sub_dict = outputs_to_write[participant][subfolder]
            if camera in sub_dict:
                for key in keys_to_store:
                    sub_dict[camera][key] = np.concatenate([sub_dict[camera][key],
                                                            outputs[key][i, :]], axis=0)
            else:
                sub_dict[camera] = {}
                for key in keys_to_store:
                    sub_dict[camera][key] = outputs[key][i, :]

            sequence_key = (participant, subfolder, camera)
            if sequence_key not in processed_so_far:
                print('Handling %s/%s/%s' % sequence_key)
                processed_so_far.add(sequence_key)

    # Write output file
    if first_round:
        model.save_subject_memory(memory_path)
        print('model memory saved', memory_path, '---ejfjdsf eval_codalab.py')
    else:
        print('memory not changed, read only', memory_path, '---vjjeiioj')
    output_fname = 'for_codalab_%s' % time.strftime('%y%m%d_%H%M%S') + '_' + output_suffix + '.pkl.gz'
    output_path = output_dir + 'eval_codalab_' + config.config_id
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    final_output_path = os.path.join(output_path, output_fname)
    print(final_output_path, '---ejfoijijd src/eval_codalab.py')
    with gzip.open(final_output_path, 'wb') as f:
        pickle.dump(outputs_to_write, f, protocol=3)
    pickle.dump(outputs_to_write, open(final_output_path + '.p', 'wb'))


