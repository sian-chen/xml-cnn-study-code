import copy
import json
import logging
import os
import time

import numpy as np
import torch
from pytorch_lightning.utilities.seed import seed_everything


class Timer(object):
    """Computes elasped time."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.running = True
        self.total = 0
        self.start = time.time()
        return self

    def resume(self):
        if not self.running:
            self.running = True
            self.start = time.time()
        return self

    def stop(self):
        if self.running:
            self.running = False
            self.total += time.time() - self.start
        return self

    def time(self):
        if self.running:
            return self.total + time.time() - self.start
        return self.total


def dump_log(log_path, metrics=None, split=None, config=None, meta=None):
    """Write log including config and the evaluation scores.

    Args:
        log_path(str): path to log path
        metrics (dict): metric and scores in dictionary format, defaults to None
        split (str): val or test, defaults to None
        config (dict): config to save, defaults to None
    """
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    if os.path.isfile(log_path):
        with open(log_path) as fp:
            result = json.load(fp)
    else:
        result = dict()

    if config:
        config_to_save = copy.deepcopy(dict(config))
        config_to_save.pop('device', None)  # delete if device exists
        result['config'] = config_to_save
    if split and metrics:
        if split in result:
            result[split].append(metrics)
        else:
            result[split] = [metrics]
    if meta:
        result['meta'] = meta

    with open(log_path, 'w') as fp:
        json.dump(result, fp)

    logging.info(f'Finish writing log to {log_path}.')


def set_seed(seed):
    """Set seeds for numpy and pytorch."""
    if seed is not None:
        if seed >= 0:
            seed_everything(seed=seed)
            torch.use_deterministic_algorithms(True)
            torch.backends.cudnn.benchmark = False
        else:
            logging.warning(
                f'the random seed should be a non-negative integer')


def init_device(use_cpu=False):
    if not use_cpu and torch.cuda.is_available():
        # Set a debug environment variable CUBLAS_WORKSPACE_CONFIG to ":16:8" (may limit overall performance) or ":4096:8" (will increase library footprint in GPU memory by approximately 24MiB).
        # https://docs.nvidia.com/cuda/cublas/index.html
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ":4096:8"
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
        # https://github.com/pytorch/pytorch/issues/11201
        torch.multiprocessing.set_sharing_strategy('file_system')
    logging.info(f'Using device: {device}')
    return device


def argsort_top_k(vals, k, axis=-1):
    unsorted_top_k_idx = np.argpartition(vals, -k, axis=axis)[:,-k:]
    unsorted_top_k_scores = np.take_along_axis(vals, unsorted_top_k_idx, axis=axis)
    sorted_order = np.argsort(-unsorted_top_k_scores, axis=axis)
    sorted_top_k_idx = np.take_along_axis(unsorted_top_k_idx, sorted_order, axis=axis)
    return sorted_top_k_idx


def pad_sequence(sequences, batch_first=False, padding_value=0.0, max_len=None):
    # type: (List[Tensor], bool, float) -> Tensor
    r"""Pad a list of variable length Tensors with ``padding_value``
    Modified from pytorch.nn.utils.rnn.pad_sequence to support length specification
    https://pytorch.org/docs/stable/_modules/torch/nn/utils/rnn.html#pad_sequence
    Args:
        sequences (list[Tensor]): list of variable length sequences.
        batch_first (bool, optional): output will be in ``B x T x *`` if True, or in
            ``T x B x *`` otherwise
        padding_value (float, optional): value for padded elements. Default: 0.
        max_len (int, optional): length to pad if given, or calculate from
			sequences otherwise. Default: 0.
    Returns:
        Tensor of size ``T x B x *`` if :attr:`batch_first` is ``False``.
        Tensor of size ``B x T x *`` otherwise
    """

    # assuming trailing dimensions and type of all the Tensors
    # in sequences are same and fetching those from sequences[0]
    max_size = sequences[0].size()
    trailing_dims = max_size[1:]
    if max_len is None:
        max_len = max([s.size(0) for s in sequences])
    if batch_first:
        out_dims = (len(sequences), max_len) + trailing_dims
    else:
        out_dims = (max_len, len(sequences)) + trailing_dims

    out_tensor = sequences[0].new_full(out_dims, padding_value)
    for i, tensor in enumerate(sequences):
        length = tensor.size(0)
        # use index notation to prevent duplicate references to the tensor
        if batch_first:
            out_tensor[i, :length, ...] = tensor
        else:
            out_tensor[:length, i, ...] = tensor

    return out_tensor
