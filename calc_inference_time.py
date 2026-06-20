#!/usr/bin/env python
"""Measure the per-utterance inference latency of each defense module."""
import time
import json
import os

import torch
from datasets import load_from_disk, Audio
from torch.utils.data import DataLoader
import torch.nn as nn
from torchaudio.functional import add_noise
from safetensors import safe_open
from df import init_df 
from torchmetrics.audio import SignalNoiseRatio
from tqdm import tqdm

from torchdf.torch_df_offline import TorchDF
from dns_mpsenet_components.generator import MPNet
from defense_modules import *
from victim_models import *
from attacks.pgd import AudioAttack




torch.cuda.set_device(1)
device = f"cuda"




class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self




def load_checkpoint(filepath, device):
    assert os.path.isfile(filepath)
    print("Loading '{}'".format(filepath))
    checkpoint_dict = torch.load(filepath, map_location=device)
    print("Complete.")
    return checkpoint_dict




# noise settings
beta_min = 0.02
beta_max = 4
N = 200
total_noise_levels=3
n_fft = 400
hop_size = 100
win_size = 400
compress_factor = 0.3




noise_module = GaussianNoiseAddingDBFS(-35)
with open("/home/apoman123/mpsenet-speech-enhanced/finetune_config.json", "r") as f:
        data = f.read()
json_config = json.loads(data)
h = AttrDict(json_config)
mpsenet = MPNet(h).to(device)
state_dict = load_checkpoint("/data/nas07/SharedBySMB/apoman123/mpsenet", device)
mpsenet.load_state_dict(state_dict['generator'])
defense_module = DNSMPSENetDefenseModule(mpsenet, noise_module, n_fft, hop_size, win_size, compress_factor)




args_dict = {
    'ddpm_path': '/home/apoman123/speech_enhancement_adversarial_evaluation/diffusion_models/unconditional_diffwave.pkl',
    'ddpm_config': '/home/apoman123/speech_enhancement_adversarial_evaluation/diffusion_models/DiffWave_Unconditional/config.json',
    't': 3,
    'sample_step': 1,
    't_delta': 0,
    'rand_t': False,
    'diffusion_type': 'ddpm',
    'score_type': 'guided_diffusion',
    'use_bm': False
            }
audio_pure_config = AttrDict(args_dict)




from diffusion_models.diffwave_sde import RevDiffWave
from defense_modules import AudioPureDefenseModule
audio_pure = RevDiffWave(audio_pure_config, device)
audio_pure = AudioPureDefenseModule(audio_pure)




wav_1s = torch.randn(1, 16000)
wav_3s = torch.randn(1, 16000*3)
wav_5s = torch.randn(1, 16000*5)
wav_10s = torch.randn(1, 16000*10)
wav_15s = torch.randn(1, 16000*15)
wav_20s = torch.randn(1, 16000*20)
wav_25s = torch.randn(1, 16000*25)
wav_30s = torch.randn(1, 16000*30)

inference_data = [wav_1s, wav_3s, wav_5s, wav_10s, wav_15s, wav_20s, wav_25s, wav_30s]




def calc_time(model, data):
    start_time = time.time()
    result = model(data)
    stop_time = time.time()
    duration = stop_time - start_time
    return duration




audiopure_total_time = []
our_total_time = []
audio_pure.eval()
defense_module.eval()
with torch.no_grad():
    for data in tqdm(inference_data, total=len(inference_data)):
        data = data.to(device)
        audiopure_total_time.append(calc_time(audio_pure, data))
        our_total_time.append(calc_time(defense_module, data))

