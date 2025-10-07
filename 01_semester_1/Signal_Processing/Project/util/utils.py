import time
import os

import torch
from pesq import pesq
from pystoi.stoi import stoi
import numpy as np
import importlib

def prepare_empty_dir(dirs: str, resume=False):
    """
    if resume experiment, assert the dirs exist,
    if not resume experiment, make dirs.

    Args:
        dirs (list): directors list
        resume (bool): whether to resume experiment, default is False
    """
    
    for dir_path in dirs:

        if resume:
                assert dir_path.exists()
        else:
                dir_path.mkdir(parents=True, exist_ok=True)

class ExecutionTime:
    """
    Usage:
    timer = ExecutionTime()
    <code>
    print(f"Finished in {timer.duration()} seconds")
    """
    def __init__(self):
        self.start_time = time.time()
    
    def duration(self):
         return int(time.time() - self.start_time)

def compute_PESQ(clean_signal, noisy_signal, sr=16000):
     return pesq(sr, clean_signal, noisy_signal, 'nb')

def compute_STOI(clean_signal, noisy_signal, sr=16000):
    return stoi(clean_signal, noisy_signal, sr, extended=False)

def z_score(m):
     
     mean= np.mean(m)
     std_var = np.std(m)
     return (m-mean)/std_var, mean, std_var

def min_max(m):
    m_max = np.max(m)
    m_min = np.min(m)

    return (m - m_min) / (m_max - m_min), m_max, m_min


def reverse_min_max(m, m_max, m_min):
    return m * (m_max - m_min) + m_min

def initialize_config(module_cfg, pass_args=True):
    """According to config items, load specific module dynamically with params.
    e.g., Config items as follow：
        module_cfg = {
            "module": "model.model",
            "main": "Model",
            "args": {...}
        }
    1. Load the module corresponding to the "module" param.
    2. Call function (or instantiate class) corresponding to the "main" param.
    3. Send the param (in "args") into the function (or class) when calling ( or instantiating)
    """
    module = importlib.import_module(module_cfg["module"])

    if pass_args:
        return getattr(module, module_cfg["main"])(**module_cfg["args"])
    else:
        return getattr(module, module_cfg["main"])
    

def load_checkpoint(checkpoint_path, device):
    _, ext = os.path.splitext(os.path.basename(checkpoint_path))
    assert ext in (".pth", ".tar"), "Only support ext and tar extensions of model checkpoint."
    model_checkpoint = torch.load(checkpoint_path, map_location=device)

    if ext == ".pth":
        print(f"Loading {checkpoint_path}.")
        return model_checkpoint
    else:  # tar
        print(f"Loading {checkpoint_path}, epoch = {model_checkpoint['epoch']}.")
        return model_checkpoint["model"]
