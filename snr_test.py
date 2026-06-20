import json
import os

import torch
from datasets import load_from_disk, Audio
from torch.utils.data import DataLoader
import torch.nn as nn
from torchattacks import PGD
from torchaudio.functional import add_noise
from safetensors import safe_open
from df import init_df 
from torchmetrics.audio import SignalNoiseRatio
from tqdm import tqdm

from torchdf.torch_df_offline import TorchDF
from dns_mpsenet_components.generator import MPNet
from defense_modules import *
from victim_models import *
torch.cuda.set_device(1)
device = f"cuda"
# torch.backends.cudnn.enabled = False

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


def collate(samples):
    input_values = [
        torchaudio.functional.vad(torch.from_numpy(sample['audio']['array']).float(), sample_rate=16000) 
        if torchaudio.functional.vad(torch.from_numpy(sample['audio']['array']).float(), sample_rate=16000).shape[-1] != 0 else torch.from_numpy(sample['audio']['array']).float()
        for sample in samples
    ]
    lens = [value.shape[-1] for value in input_values]
    max_len = max(lens)
    input_values = [
        torch.cat([
            value,
            torch.zeros(max_len - value.shape[-1])
        ], dim=-1) for value in input_values
    ]

    labels = torch.tensor([sample['label'] for sample in samples])
    
    return {'input_values': torch.stack(input_values, dim=0), 'labels': labels} # cut to 3 seconds

vctk = load_from_disk("/data/nas07/PersonalData/apoman123/vctk_mfcc_with_speaker_labels").cast_column('audio', Audio(sampling_rate=16000))
vctk = vctk.shuffle(seed=42).train_test_split(test_size=0.2, shuffle=False)
sc09 = load_from_disk("/data/nas07/SharedBySMB/apoman123/sc09").cast_column('audio', Audio(sampling_rate=16000))
# esc50_filtered = load_from_disk("/data/nas07/PersonalData/apoman123/esc50_rain")
qkws = load_from_disk("/data/nas07/PersonalData/apoman123/QKWS").cast_column('audio', Audio(sampling_rate=16000))
qkws_loader = DataLoader(qkws['test'], batch_size=1, collate_fn=collate, num_workers=10)
sc09_loader = DataLoader(sc09['test'], batch_size=1, collate_fn=collate, num_workers=10)
vctk_loader = DataLoader(vctk['test'], batch_size=1, collate_fn=collate, num_workers=10)

# noise settings
beta_min = 0.02
beta_max = 4
N = 200
total_noise_levels=3
n_fft = 400
hop_size = 100
win_size = 400
compress_factor = 0.3

# noise_module = VPSDENoiseAdding(beta_min, beta_max, N, total_noise_levels)
# noise_module = BackgroundNoiseAdding([torch.from_numpy(data['audio']['array']).float() for data in esc50_filtered])
# noise_module = Identity()
noise_module = GaussianNoiseAdding(0)
with open("/home/apoman123/mpsenet-speech-enhanced/finetune_config.json", "r") as f:
        data = f.read()
json_config = json.loads(data)
h = AttrDict(json_config)
mpsenet = MPNet(h).to(device)
state_dict = load_checkpoint("/data/nas07/SharedBySMB/apoman123/mpsenet", device)
mpsenet.load_state_dict(state_dict['generator'])
defense_module = DNSMPSENetDefenseModule(mpsenet, noise_module, n_fft, hop_size, win_size, compress_factor)

# victim model setting
# victim = M5(n_input=1, n_output=10)
# victim = X_vector(60, 108)
victim = KWSModel()
# transform = torchaudio.transforms.MFCC(16000, 40).to(device)
# transform = Xvector_Features()
transform = RCNN_Features(zero_mean=True)
classifier_weights = {}
with safe_open("/data/nas07/PersonalData/apoman123/rcnn_zero_mean_qkws/checkpoint-5900/model.safetensors", framework="pt") as f:
    for key in f.keys():
        classifier_weights[key] = f.get_tensor(key)
victim.load_state_dict(classifier_weights)
# victim.load_state_dict(torch.load("/data/nas07/PersonalData/apoman123/m5_sc09/m5net_66.pt", map_location="cpu"))
victim.to(device)

classifier = Classifier(victim, transform)
classifier.to(device)

model = AcousticSystem(defense_module, classifier)
model.to(device)

def test_snr(defense_module, loader):
    snr_metric = SignalNoiseRatio()
    result_snrs = {}
    with torch.no_grad():
        for snr in tqdm(range(-5, 10)):
            defense_module.noise_adding.snr = snr
            generated_snrs = []
            for batch in tqdm(loader, total=len(loader)):
                generated = defense_module(batch['input_values'].to(device)).cpu()
                batch['input_values'] = batch['input_values'][:, :generated.shape[-1]]
                generated_snrs.append(snr_metric(generated, batch['input_values']))
            
            mean_snr = torch.stack(generated_snrs, dim=0).mean()
            result_snrs[f'{snr}'] = mean_snr
    return result_snrs

qkws_snrs = test_snr(defense_module, qkws_loader)
vctk_snrs = test_snr(defense_module, vctk_loader)
sc09_snrs = test_snr(defense_module, sc09_loader)

print(f"here is qkws result: {qkws_snrs}")
print(f"here is sc09 result: {sc09_snrs}")
print(f"here is vctk result: {vctk_snrs}")