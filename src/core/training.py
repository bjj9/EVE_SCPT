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

import argparse
from collections import OrderedDict
import functools
import gc
import hashlib
import logging
import os
import sys
import time
import psutil

import coloredlogs
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from core import DefaultConfig, CheckpointManager, Tensorboard

config = DefaultConfig() # wfhuefh training.py

# Setup logger
logger = logging.getLogger(__name__)

# Set device
# device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
curr_device_index = torch.cuda.current_device()
#device = torch.device('cuda:'+str(curr_device_index)  if torch.cuda.is_available() else "cpu")
device = torch.device(("cuda:0" if config.use_one_gpu is None else 'cuda:'+str(config.use_one_gpu) )  if torch.cuda.is_available() else "cpu")
device = torch.device('cuda:0')
print(device, '---oigfedfd training.py')

def _convert_cli_arg_type(key, value):
    config_type = type(getattr(config, key))
    if config_type == bool:
        if value.lower() in ('true', 'yes', 'y') or value == '1':
            return True
        elif value.lower() in ('false', 'no', 'n') or value == '0':
            return False
        else:
            raise ValueError('Invalid input for bool config "%s": %s' % (key, value))
    else:
        return config_type(value)


def script_init_common(default_model_config):

    parser = argparse.ArgumentParser(description='Train a gaze estimation model.')
    parser.add_argument('-v', type=str, help='Desired logging level.', default='info',
                        choices=['debug', 'info', 'warning', 'error', 'critical'])
    parser.add_argument('config_json', type=str, nargs='*', help=('Path to config in JSON format. '
                              'Multiple configs will be parsed in the specified order.'))
    for key in dir(config): # nwifjijej training.py
        if key.startswith('_DefaultConfig') or key.startswith('__'):
            continue
        if key in vars(DefaultConfig) and isinstance(vars(DefaultConfig)[key], property):
            continue
        value = getattr(config, key)
        value_type = type(value)
        arg_type = value_type
        if value_type == bool:
            # Handle booleans separately, otherwise arbitrary values become `True`
            arg_type = str
        if callable(value):
            continue
        parser.add_argument('--' + key.replace('_', '-'), type=arg_type, metavar=value,
                            help='Expected type is `%s`.' % value_type.__name__)
    args = parser.parse_args()

    # Set device
    curr_device_index = torch.cuda.current_device()
    # device = torch.device('cuda:' + str(curr_device_index) if torch.cuda.is_available() else "cpu")
    # device = torch.device(("cuda:0" if config.use_one_gpu is None else 'cuda:'+str(config.use_one_gpu) )  if torch.cuda.is_available() else "cpu")

    # Set logger format and verbosity level
    coloredlogs.install(
        datefmt='%d/%m %H:%M:%S',
        fmt='%(asctime)s %(levelname)s %(message)s',
        level=args.v.upper(),
    )

    # Parse configs in order specified by user
    if default_model_config:
        training_defaul_json_dir = default_model_config
        print('before json import default_model_config configurations',
              'batch_size', config.batch_size, 'config id', config.config_id, 'output_dir', config.output_dir, '---neagfsfdf')
        config.import_json(training_defaul_json_dir) # bwbiufhjef training.py
        print('import configuration in', training_defaul_json_dir , 'to cover config_default, as new default in training.py and inference.py', '---ijodjoijfe')
        print('after json import default_model_config configurations', 'batch_size',
              config.batch_size, 'config id', config.config_id, 'output_dir', config.output_dir, '---enreagfrfddf')

    for json_path in args.config_json:
        print('before json import args.config_json',
              'batch_size', config.batch_size, 'config id', config.config_id, 'resume_from', config.resume_from, 'output_dir', config.output_dir, '---nrgewwf')
        config.import_json(json_path)
        print('after json import args.config_json',
              'batch_size', config.batch_size, 'config id', config.config_id, 'resume_from', config.resume_from, 'output_dir', config.output_dir, '---aegvwefdwef')

    # Apply configs passed through command line
    config.import_dict({
        key.replace('-', '_'): _convert_cli_arg_type(key, value)
        for key, value in vars(args).items()
        if value is not None and hasattr(config, key)
    })

    # Improve reproducibility
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    if config.fully_reproducible:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    np.random.seed(0)

    #print('\nconfig', config.get_all_key_values(), '---efjjsjsj training.py')
    print('final configurations', 'batch_size', config.batch_size,
          'config id', config.config_id, 'resume_from', config.resume_from, config.refine_net_enabled, 'output_dir', config.output_dir, '---eafefwefr training.py')
    return config


def init_datasets(train_specs, test_specs):

    # Initialize training datasets
    train_data = OrderedDict()
    for tag, dataset_class, path, stimuli, cameras in train_specs:
        dataset = dataset_class(path,
                                cameras_to_use=cameras,
                                types_of_stimuli=stimuli)
        dataset.original_full_dataset = dataset
        dataloader = DataLoader(dataset,
                                batch_size=config.batch_size,
                                shuffle=True,
                                drop_last=True,
                                num_workers=config.train_data_workers,
                                pin_memory=True,
                                )
        train_data[tag] = {
            'dataset': dataset,
            'dataloader': dataloader,
        }
        logger.info('> Ready to use training dataset: %s' % tag)
        logger.info('          with number of videos: %d' % len(dataset))

    # Initialize test datasets
    test_data = OrderedDict()
    for tag, dataset_class, path, stimuli, cameras in test_specs:
        # Get the full dataset
        dataset = dataset_class(path,
                                cameras_to_use=cameras,
                                types_of_stimuli=stimuli,
                                live_validation=True)
        dataset.original_full_dataset = dataset
        # then subsample datasets for quicker testing
        num_subset = config.test_num_samples
        if len(dataset) > num_subset:
            subset = Subset(dataset, sorted(np.random.permutation(len(dataset))[:num_subset]))
            subset.original_full_dataset = dataset
            dataset = subset
        dataloader = DataLoader(dataset,
                                batch_size=config.test_batch_size,
                                shuffle=False,
                                num_workers=config.test_data_workers,
                                pin_memory=True,
                                )
        test_data[tag] = {
            'dataset': dataset,
            'dataset_class': dataset_class,
            'dataset_path': path,
            'dataloader': dataloader,
        }
        logger.info('> Ready to use evaluation dataset: %s' % tag)
        logger.info('           with number of entries: %d' % len(dataset.original_full_dataset))
        if dataset.original_full_dataset != dataset:
            logger.info('     of which we evaluate on just: %d' % len(dataset))
    return train_data, test_data

def setup_common(model, optimizers):
    identifier = (model.__class__.__name__ +
                  config.identifier_suffix + '/' +
                  time.strftime('%y%m%d_%H%M%S') + '.' +
                  hashlib.md5(config.get_full_json().encode('utf-8')).hexdigest()[:6] + '.' +
                  config.training_version
                  ) # wjeoijoiji training.py

    if len(config.resume_from) > 0:
        identifier = '/'.join(config.resume_from.split('/')[-2:])
        output_dir = config.resume_from
    else:
        output_dir = '../outputs/' + identifier

    print(identifier, output_dir, '---noiwjfioj')

    # Initialize tensorboard
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    tensorboard = Tensorboard(output_dir)

    # Write source code to output dir
    # NOTE: do not over-write if resuming from an output directory
    if len(config.resume_from) == 0:
        config.write_file_contents(output_dir)

    # Log messages to file
    root_logger = logging.getLogger()
    file_handler = logging.FileHandler(output_dir + '/messages.log')
    file_handler.setFormatter(root_logger.handlers[0].formatter)
    for handler in root_logger.handlers[1:]:  # all except stdout
        root_logger.removeHandler(handler)
    root_logger.addHandler(file_handler)

    # Print model details
    num_params = sum([
        np.prod(p.size())
        for p in filter(lambda p: p.requires_grad, model.parameters())
    ])
    logger.info('\nThere are %d trainable parameters.\n' % num_params)

    # Cache base and target learning rate for each optimizer
    for optimizer in optimizers:
        optimizer.target_lr = optimizer.param_groups[0]['lr']
        optimizer.base_lr = optimizer.target_lr / config.batch_size

    # Sneak in some extra information into the model class instance
    model.identifier = identifier
    model.output_dir = output_dir
    model.checkpoint_manager = CheckpointManager(model, optimizers)
    # model.gsheet_logger = GoogleSheetLogger(model)
    model.last_epoch = 0.0
    model.last_step = 0

    # Load pre-trained model weights if available
    if len(config.resume_from) > 0:
        model.last_step = model.checkpoint_manager.load_last_checkpoint()

    return model, optimizers, tensorboard


def salvage_memory():
    """Try to free whatever memory that can be freed."""
    torch.cuda.empty_cache()
    gc.collect()


def get_training_batches(train_data_dicts):
    """Get training batches of data from all training data sources."""
    out = {}
    for tag, data_dict in train_data_dicts.items():
        if 'data_iterator' not in data_dict:
            data_dict['data_iterator'] = iter(data_dict['dataloader'])
        # Try to get data
        while True:
            try:
                # p0 = time.time()
                out[tag] = next(data_dict['data_iterator'])
                # p1 = time.time()
                # print('get_training_batches()', tag, 'costs', round(p1 - p0, 3), '---miejiojio')
                break
            except StopIteration:
                print('get_training_batches() failed to get', tag, '---nbijfjefu')
                del data_dict['data_iterator']
                salvage_memory()
                data_dict['data_iterator'] = iter(data_dict['dataloader'])

        # Move tensors to GPU
        for k, v in out[tag].items():
            if isinstance(v, torch.Tensor):
                out[tag][k] = v.detach()
                if k != 'screen_full_frame':
                    if not config.multi_gpu:
                        out[tag][k] = out[tag][k].to(device, non_blocking=True)
                    else:
                        out[tag][k] = out[tag][k].cuda(non_blocking=True)
            else:
                out[tag][k] = v
    return out


def test_model_on_all(model, test_data_dicts, current_step, tensorboard=None, log_key_prefix='test'):
    """Get training batches of data from all training data sources."""
    model.eval()
    salvage_memory()
    final_out = {}
    for tag, data_dict in test_data_dicts.items():
        with torch.no_grad():
            num_entries = len(data_dict['dataset'])
            for i, input_data in enumerate(data_dict['dataloader']):
                batch_size = next(iter(input_data.values())).shape[0]

                # Move tensors to GPU
                for k, v in input_data.items():
                    if isinstance(v, torch.Tensor):
                        if not config.multi_gpu:
                            input_data[k] = v.detach().to(device, non_blocking=True)
                        else:
                            input_data[k] = v.detach().cuda(non_blocking=True)
                # Inference
                batch_out = model(input_data, create_images=(i == 0))
                weighted_batch_out = dict([
                    (k, v.detach().cpu().numpy() * (batch_size / num_entries))
                    for k, v in batch_out.items() if v.dim() == 0
                ])
                if tag not in final_out:
                    final_out[tag] = dict([(k, 0.0) for k in weighted_batch_out.keys()])
                for k, v in weighted_batch_out.items():
                    final_out[tag][k] += v

                # Log images
                if i == 0:
                    assert tensorboard
                    if 'images' in batch_out:
                        import torchvision.utils as vutils
                        tensorboard.add_image(
                            log_key_prefix + '_%s/images' % tag,
                            vutils.make_grid(batch_out['images'].detach()[:8, :],
                                             nrow=1,  # One entry per row
                                             padding=20,
                                             normalize=True,
                                             scale_each=True,
                                             )
                        )

        # Calculate mean error over whole dataset
        logger.info('%10s test: %s' % ('[%s]' % tag,
                                       ', '.join(['%s: %.4g' % (k, final_out[tag][k])
                                                  for k in sorted(final_out[tag].keys())])))

        # Write to tensorboard
        if tensorboard:
            tensorboard.update_current_step(current_step)
            for k, v in final_out[tag].items():
                tensorboard.add_scalar(log_key_prefix + '_%s/%s' % (tag, k), v)

    # Log training metrics to Google Sheets
    for_gsheet = None
    # if model.gsheet_logger.ready:
    #     for_gsheet = {}
    #     for tag, out in final_out.items():
    #         for k, v in out.items():
    #             for_gsheet[log_key_prefix + '/%s/%s' % (tag, k)] = v

    # Free up memory
    salvage_memory()

    return final_out, for_gsheet


def do_final_full_test(model, test_data, tensorboard):
    previously_registered_dataset_classes = {}
    for k, v in test_data.items():
        # Get the full dataset
        if 'dataloader' in test_data[k]:
            del v['dataloader']
        full_original_dataset = v['dataset'].original_full_dataset
        previously_registered_dataset_classes[k] = v['dataset']
        new_dataset = v['dataset_class'](
            v['dataset_path'],
            cameras_to_use=full_original_dataset.cameras_to_use,
            types_of_stimuli=full_original_dataset.types_of_stimuli,
            is_final_test=True,
        )
        test_data[k]['dataset'] = new_dataset
        test_data[k]['dataloader'] = DataLoader(new_dataset,
                                                batch_size=config.full_test_batch_size,
                                                shuffle=False,
                                                num_workers=config.full_test_data_workers,
                                                pin_memory=True,
                                                )
        logger.info('> Ready to do full test on dataset: %s' % k)
        logger.info('          with number of sequences: %d' % len(new_dataset))

    logger.info('# Now beginning full test on all evaluation sets.')
    logger.info('# Hold on tight, this might take a while.')
    logger.info('#')
    final_out, for_gsheet = test_model_on_all(model, test_data, model.last_step + 2,
                                      tensorboard=tensorboard,
                                      log_key_prefix='full_test')

    # Restore dataset class
    for k, v in test_data.items():
        test_data[k]['dataset'] = previously_registered_dataset_classes[k]

    # Clean up dataloaders
    for k, v in test_data.items():
        del v['dataloader']

    # Log training metrics to Google Sheets
    if for_gsheet is not None:
        model.gsheet_logger.update_or_append_row(for_gsheet)

    # Free memory
    salvage_memory()


def learning_rate_schedule(optimizer, epoch_len, tensorboard_log_func, step):
    num_warmup_steps = int(epoch_len * config.num_warmup_epochs)
    selected_lr = None
    if step < num_warmup_steps:
        b = optimizer.base_lr
        a = (optimizer.target_lr - b) / float(num_warmup_steps)
        selected_lr = a * step + b
    else:
        # Decay learning rate with step function and exponential decrease?
        new_step = step - num_warmup_steps
        epoch = new_step / float(epoch_len)
        current_interval = int(epoch / config.lr_decay_epoch_interval)
        if config.lr_decay_strategy == 'exponential':
            # Step function decay
            selected_lr = optimizer.target_lr * np.power(config.lr_decay_factor, current_interval)
        elif config.lr_decay_strategy == 'cyclic':
            # Note, we start from the up state (due to previous warmup stage)
            # so each period consists of down-up (not up-down)
            peak_a = optimizer.target_lr * np.power(config.lr_decay_factor, current_interval)
            peak_b = peak_a * config.lr_decay_factor
            half_interval = 0.5 * config.lr_decay_epoch_interval
            current_interval_start = current_interval * config.lr_decay_epoch_interval
            current_interval_half = current_interval_start + half_interval
            if epoch < current_interval_half:
                # negative slope (down from peak_a)
                slope = -(peak_a - optimizer.base_lr) / half_interval
            else:
                # positive slope (up to peak_b)
                slope = (peak_b - optimizer.base_lr) / half_interval
            selected_lr = slope * (epoch - current_interval_half) + optimizer.base_lr
        else:
            selected_lr = optimizer.target_lr

    # Log to Tensorboard and return
    if step_modulo(step, config.tensorboard_learning_rate_every_n_steps):
        tensorboard_log_func(selected_lr)
    return selected_lr


def step_modulo(current, interval_size):
    return current % interval_size == (interval_size - 1)
    # return current % interval_size == 0


def main_loop_iterator(model, optimizers, train_data, test_data, tensorboard=None, do_before_forward_pass=None):
    # Skip this entirely if requested
    if config.skip_training:
        return
    print('main loop entered', config.skip_training, tensorboard is None, list(train_data.keys()), '---jjjieiiid')
    assert tensorboard is not None  # We assume this exists in LR schedule logging
    initial_step = model.last_step  # Allow resuming
    max_dataset_len = np.amax([len(data_dict['dataset']) for data_dict in train_data.values()])
    num_steps_per_epoch = int(max_dataset_len / config.batch_size)
    num_training_steps = int(config.num_epochs * num_steps_per_epoch)
    lr_schedulers = [
        torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            functools.partial(learning_rate_schedule, optimizer, num_steps_per_epoch,
                              functools.partial(tensorboard.add_scalar, 'lr/optim_%d' % i)),
        ) for i, optimizer in enumerate(optimizers)
    ]
    model.train()
    current_step = 0
    # training_result_list = []
    print('main_loop_iterator', max_dataset_len, num_steps_per_epoch, 'starts training from', initial_step, 'to', num_training_steps, '---nvjjieijj')
    p_training_start = time.time()
    for current_step in range(initial_step, num_training_steps):
        p0 = time.time()
        # print('main_loop_iterator() enter training loop', current_step, '---jiojijfj')
        current_epoch = (current_step * config.batch_size) / max_dataset_len  # fractional value
        tensorboard.update_current_step(current_step + 1)
        # print('main_loop_iterator() enter training loop, snap before get_training_batches(train_data)', current_step, current_epoch, '---uhgujfeij')
        #monitor_ram(landmark='before loading input data ---fhwiuhieu')
        pdata0 = time.time()
        input_data = get_training_batches(train_data)
        pdata1 = time.time()
        # print('get_training_batches()', tag, 'costs', round(pdata1 - pdata0, 3), '---miejiojio')
        #monitor_ram(landmark='step ' + str(current_step) + ' after loading input data ---ncounjfed')

        # Set correct states before training iteration
        # print('main_loop_iterator() enter training loop, snap before model.train()', 'to be added', '---nbfjiej')
        model.train()
        # print('main_loop_iterator() enter training loop, snap after model.train()', '---waragff')
        for optimizer in optimizers:
            optimizer.zero_grad()

        # If routine defined for before forward pass, do it
        if do_before_forward_pass:
            do_before_forward_pass(current_step)

        # Prepare keyword arguments to model inference


        # Forward pass and yield
        loss_terms = []
        images_to_log_to_tensorboard = {}
        pp0 = time.time()


        # model.forward_kwargs = forward_kwargs
        # forward_kwargs = {
        #     'create_images': step_modulo(current_step, config.tensorboard_images_every_n_steps),
        #     'current_epoch': current_epoch,
        # }
        create_images = step_modulo(current_step, config.tensorboard_images_every_n_steps)
        # print(device, current_epoch, create_images, '---njfiejffj training.py')

        pp1 = time.time()
        outputs = model(input_data, current_epoch=current_epoch, create_images=create_images)
        # if not config.multi_gpu:
        #     model.forward_kwargs = forward_kwargs
        #     outputs = model(input_data)
        # else:
        #     device_ids = config.device_ids
        #     output_device = device_ids[0]
        #     model.forward_kwargs = forward_kwargs #TODO: add forward keyword arguments as model attribute
        #     replicas = nn.parallel.replicate(model, device_ids)
        #
        #     #input_all = (input_data, **forward_kwargs)
        #     inputs = nn.parallel.scatter(input_data, device_ids)
        #
        #
        #     replicas = replicas[:len(inputs)]
        #     print([type(r) for r in replicas], '---bjefeoijij')
        #     print(model.peek_dict(input_data, '---efjij'), '---nnjjiejfi')
        #     for ii, inp in enumerate(inputs):
        #         print('input', ii, model.peek_dict(inp, '---nxxxxj'), '---w0fijrifj')
        #     #raise
        #
        #     ta = nn.parallel.parallel_apply(replicas, inputs)
        #     outputs = nn.parallel.gather(ta, output_device)

        torch.cuda.device_count()

        # print('main_loop_iterator() enter training loop,', 'snap before yield', loss_terms, '    |||||||     ', outputs, '---jiwejfi')
        # Time monitoring
        p1 = time.time()

        # print('get_training_batches()', tag, 'costs', round(pdata1 - pdata0, 3), '---miejiojio')
        # print('step', current_step, 'model forward costs', round(p1 - pp0, 3), '---nbfhujeij')
        time_cost_this_step = p1 - p0
        time_cost_till_now = p1 - p_training_start
        steps_remaining = num_training_steps - current_step
        time_remaining = time_cost_till_now / (current_step - initial_step + 1) * steps_remaining
        time_cost_fetch_batch = pdata1 - pdata0
        time_cost_model_forward = p1 - pp0

        ## store results aside tensorboard
        # outputs_key_str = 'losses_and_metrics'
        # if outputs_key_str not in model.checkpoint_manager.history_stats.keys(): model.checkpoint_manager.history_stats[outputs_key_str] = {}
        # model.checkpoint_manager.history_stats[outputs_key_str][current_step] = outputs
        # training_result_list.append((current_step, outputs.items()))
        # print('loss term size before yield', len(loss_terms), loss_terms, '---jgqwewr')
        # print('monitor history_stats()', current_step, list(model.checkpoint_manager.history_stats['losses_and_metrics'].keys()), '---nwfiuuiuii')

        yield current_step, loss_terms, outputs, images_to_log_to_tensorboard, time_cost_this_step, steps_remaining, time_remaining, time_cost_fetch_batch, time_cost_model_forward

        print('step', current_step, 'costs time', round(time_cost_this_step, 2),
              '(get_data_batch()', round(time_cost_fetch_batch, 2), '+ forward()', round(time_cost_model_forward, 2), round(pp1 - pp0, 2), round(p1 - pp1, 2), ')    ',
              'steps remaining:', steps_remaining, ', time remaining:', round(time_remaining/3600, 1),
              'hours', '---gerefrf training.py') # ; loss:', outputs['full_loss'], '---kfijhguehf')

        # There should be as many loss terms as there are optimizers!
        assert len(loss_terms) == len(optimizers)
        # print('loss term size after yield (loss[full_loss] already added in train.py)', len(loss_terms), loss_terms, '---j23rwefij')

        # Prune out None values
        valid_loss_terms = []
        valid_optimizers = []
        for loss_term, optimizer in zip(loss_terms, optimizers):
            if loss_term is not None:
                #print('loss term', loss_term, '---jjejjieii training.py')
                valid_loss_terms.append(loss_term)
                valid_optimizers.append(optimizer)
        # print('main_loop_iterator() proceed y1.3', valid_loss_terms, valid_optimizers, '---ergrgr')
        # Perform gradient calculations for each loss term
        for i, (loss, optimizer) in enumerate(zip(valid_loss_terms, valid_optimizers)):
            not_last = i < (len(optimizers) - 1)
            # print('main_loop_iterator() proceed y1.4', i, (loss, optimizer), not_last, isinstance(loss, torch.Tensor), '---bewfdsd')
            if not isinstance(loss, torch.Tensor):
                continue
            # print('main_loop_iterator() proceed y1.45', type(loss), loss, '---mmiejji')
            #monitor_gpu_usage(landmark='before backward ---wnfjeifij')
            #monitor_ram(landmark='before backward ---agwwerwerr')
            # print('loss size', loss.size(), loss, loss.ndim, '---jwejijijj training.py')
            if loss.ndim > 0: # if it's not a scaler
                loss = torch.mean(loss)
            print('loss size', loss, loss.ndim, '---vehuj training.py')
            loss.backward(retain_graph=not_last)
            #monitor_gpu_usage(landmark='after backward ---jejfijioj')
            #monitor_ram(landmark='after backward ---vubiniowf')
            # print('main_loop_iterator() proceed y1.5', '---bewfdsd')
        # print('main_loop_iterator() proceed y1.6', '---jafsdf')
        # Maybe clip gradients
        if config.do_gradient_clipping:
            if config.gradient_clip_by == 'norm':
                clip_func = nn.utils.clip_grad_norm_
            elif config.gradient_clip_by == 'value':
                clip_func = nn.utils.clip_grad_value_
            clip_amount = config.gradient_clip_amount
            clip_func(model.parameters(), clip_amount)
        # print('main_loop_iterator() proceed y2', '---gwfsfs')
        # Apply gradients
        for optimizer in valid_optimizers:
            optimizer.step()

        # Print outputs
        if step_modulo(current_step, config.log_every_n_steps):
            metrics = dict([(k, torch.mean(v).detach().cpu().numpy())
                            for k, v in outputs.items()
                            if v.dim() == 0]) # wifjiojwjejfj training.py
            # seave training results
            model.checkpoint_manager.save_at_step_training_results(current_step + 1, metrics)  # jwijoijfeijff training.py
            for i, loss in enumerate(loss_terms):  # Add loss terms
                if loss is not None:
                    metrics['loss_%d' % (i + 1)] = torch.mean(loss).detach().cpu().numpy()
            # if (current_step + 1 < 10) or (current_step + 1 % 10 == 0):
            # log = ('Step %d, Epoch %.2f> ' % (current_step + 1, current_epoch)
            #        + ', '.join(['%s: %.4g' % (k, metrics[k]) for k in sorted(metrics.keys())]))
            log = ('Step %d, Epoch %.2f> ' % (current_step + 1, current_epoch)  # nvoijefoij
                   + ', '.join(['%s: %.4g' % (k, metrics[k]) for k in
                        ['full_loss', 'metric_ang_g_initial', 'metric_ang_g_final', 'metric_bce_left_tracking_validity', 'metric_bce_right_tracking_validity']
                                if k in metrics.keys()]))
            # log = ('Step %d, Epoch %.2f> ' % (current_step + 1, current_epoch)  # nvoijefoij
            #        + ', '.join(['%s: %.4g' % (k, metrics[k]) for k in ['full_loss', 'metric_ang_g_initial', 'metric_ang_g_final'] if k in metrics.keys()]))
            logger.info(log)
            # print(metrics, '---jjiejjjdkkaa training.py')
            '''
            log = ('Step %d, Epoch %.2f> ' % (current_step + 1, current_epoch)
                   + ', '.join(['%s: %.4g' % (k, metrics[k]) for k in sorted(metrics.keys()) if k in ['full_loss', 'metric_ang_g_initial', 'metric_ang_g_final']]))
            
            # all loss and metric terms:
            full_loss: 0.7607, loss_1: 0.7607, loss_ang_left_g_initial: 1.663, loss_ang_right_g_initial: 1.714, loss_ce_heatmap_final: 0.692,
            loss_ce_heatmap_initial: 0.1284, loss_l1_left_pupil_size: 0.1755, loss_l1_right_pupil_size: 0.153, loss_mse_PoG_cm_final: 68.71,
            loss_mse_PoG_cm_initial: 9.655, loss_mse_PoG_px_final: 8.283e+04, loss_mse_PoG_px_initial: 1.16e+04, loss_mse_heatmap_final: 0.2409,
            loss_mse_left_PoG_cm_initial: 4.973, loss_mse_lr_consistency: 21.69, loss_mse_right_PoG_cm_initial: 5.625, metric_ang_g_final: 8.964,
            metric_ang_g_initial: 3.18, metric_ang_g_initial_unaugmented: 1.579, metric_euc_PoG_cm_final: 9.902, metric_euc_PoG_cm_initial: 3.631,
            metric_euc_PoG_cm_initial_unaugmented: 1.764, metric_euc_PoG_px_final: 343.8, metric_euc_PoG_px_initial: 125.8,
            metric_euc_PoG_px_initial_unaugmented: 61.25, metric_euc_left_PoG_cm_initial: 1.875, metric_euc_left_PoG_px_initial: 167.3,
            metric_euc_lr_consistency: 6.32, metric_euc_right_PoG_cm_initial: 1.911, metric_euc_right_PoG_px_initial: 159'''

            # Log to Tensorboard
            if step_modulo(current_step, config.tensorboard_scalars_every_n_steps):
                for key, metric in metrics.items():
                    if key.startswith('loss_'):
                        key = key[len('loss_'):]
                        tensorboard.add_scalar('train_losses/%s' % key, metric)
                    elif key.startswith('metric_'):
                        key = key[len('metric_'):]
                        tensorboard.add_scalar('train_metrics/%s' % key, metric)
                    else:
                        tensorboard.add_scalar('train/%s' % key, metric)

                tensorboard.add_scalar('lr/epoch', current_epoch)

                if step_modulo(current_step, config.tensorboard_images_every_n_steps):
                    for k, img in images_to_log_to_tensorboard.items():
                        tensorboard.add_image(k, img)

            # Quit if NaNs
            there_are_NaNs = False
            for k, v in metrics.items():
                if np.any(np.isnan(v)):
                    logger.error('NaN encountered during training at value: %s' % k)
                    there_are_NaNs = True
            if there_are_NaNs:
                cleanup_and_quit(train_data, test_data, tensorboard)

        # We're done with the previous outputs
        # print(metrics, '---waefsfds training.py')
        del input_data, outputs, loss_terms, images_to_log_to_tensorboard

        # Save checkpoint
        if step_modulo(current_step, config.checkpoints_save_every_n_steps):
            model.checkpoint_manager.save_at_step(current_step + 1)
            # model.checkpoint_manager.save_at_step_training_results(current_step + 1, training_result_list)
            # del training_result_list
            # training_result_list = []

        # print('main_loop_iterator() proceed y3', '---wefwe')

        # Full test over all evaluation datasets
        if step_modulo(current_step, config.test_every_n_steps):
            # Do test on subset of validation datasets
            final_out, for_gsheet = test_model_on_all(model, test_data, current_step + 1, tensorboard=tensorboard)
            # store results aside tensorboard
            model.checkpoint_manager.save_at_step_testing_results(current_step + 1, final_out)
            # outputs_key_str = 'intermediate_test_results'
            # if outputs_key_str not in model.checkpoint_manager.history_stats.keys(): model.checkpoint_manager.history_stats[outputs_key_str] = {}
            # model.checkpoint_manager.history_stats[outputs_key_str][current_step] = final_out

            # Log training metrics to Google Sheets
            if for_gsheet is not None:
                for_gsheet['Step'] = current_step + 1
                for_gsheet['Epoch'] = current_epoch
                for k, v in metrics.items():
                    for_gsheet['train/' + k] = v
                # model.gsheet_logger.update_or_append_row(for_gsheet)

            # Free memory
            salvage_memory()

        # Remember what the last step/epoch were
        model.last_epoch = current_epoch
        model.last_step = current_step

        # Update learning rate
        # NOTE: should be last
        tensorboard.update_current_step(current_step + 2)
        for lr_scheduler in lr_schedulers:
            lr_scheduler.step(current_step + 1)
        # dprint('main_loop_iterator() proceed y4', '---erafdfsd')

    # We're out of the training loop now, make a checkpoint
    current_step += 1
    model.checkpoint_manager.save_at_step(current_step + 1)
    model.checkpoint_manager.save_at_step(current_step + 1, is_final_step=True)

    # Close all dataloaders
    for k, v in list(train_data.items()) + list(test_data.items()):
        if 'data_iterator' in v:
            v['data_iterator'].__del__()
            del v['data_iterator']
        v['dataloader']
        del v['dataloader']

    # Clear memory where possible
    salvage_memory()


def eval_loop_iterator(model, dataset, dataloader, create_images=False):
    """Iterate through and evaluate for a dataset."""
    model.eval()
    salvage_memory()
    with torch.no_grad():
        # num_entries = len(dataset)
        for current_step, input_data in enumerate(dataloader):
            # batch_size = next(iter(input_data.values())).shape[0]

            # Move tensors to GPU
            input_data_gpu = {}
            for k, v in input_data.items():
                if isinstance(v, torch.Tensor):
                    if not config.multi_gpu:
                        input_data_gpu[k] = v.detach().to(device, non_blocking=True)
                    else:
                        input_data_gpu[k] = v.detach().cuda(non_blocking=True)

            forward_kwargs = {
                'create_images': create_images,
            }

            # Forward pass and yield
            outputs = model(input_data_gpu, **forward_kwargs)
            yield current_step, input_data, outputs

    # Free up memory
    salvage_memory()


def cleanup_and_quit(train_data, test_data, tensorboard):
    # Close tensorboard
    if tensorboard:
        tensorboard.__del__()

    # Close all dataloaders and datasets
    for k, v in list(train_data.items()) + list(test_data.items()):
        if 'data_iterator' in v:
            v['data_iterator'].__del__()
        # if 'dataset' in v:
        #     v['dataset'].original_full_dataset.__del__()
        for item in ['data_iterator', 'dataloader', 'dataset']:
            if item in v:
                del v[item]

    # Finally exit
    sys.exit(0)

def monitor_gpu_usage(device_num=0, landmark=''):
    t = torch.cuda.get_device_properties(device_num).total_memory
    c = torch.cuda.memory_cached(device_num)
    a = torch.cuda.memory_allocated(device_num)
    f = c-a  # free inside cache
    scaleGB = lambda x: round(x/1000000000, 2) # GB
    print('GPU memory usage', '--', 'total:', scaleGB(t), ', cached:', scaleGB(c), ', allocated:', scaleGB(a), ', free:', scaleGB(f), landmark)

def monitor_ram(landmark=''):
    # import psutil
    di = dict(psutil.virtual_memory()._asdict())
    del di['percent']
    ta = sorted([(k, round(v/1000/1000/1000, 2)) for k,v in di.items()], key=lambda item: item[0])
    print('Ram usage --', ta, landmark)
    # ta['percent_used'] = psutil.virtual_memory().percent
    return ta

