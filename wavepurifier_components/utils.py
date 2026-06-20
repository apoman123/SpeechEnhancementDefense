import argparse
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import numpy as np
import torch
import torchvision.transforms as T
import io
import librosa
import numpy as np


def get_img(path):
    from PIL import Image

    # resp = requests.get('https://sparrow.dev/assets/img/cat.jpg')
    img = Image.open(path)

    preprocess = T.Compose([
        # T.Resize(256),
        # T.CenterCrop(256),
        T.ToTensor(),
        # T.Normalize(
        #     mean=[0.485, 0.456, 0.406],
        #     std=[0.229, 0.224, 0.225]
        # )
    ])

    x = preprocess(img)
    return x


def audio2wav(path):
    y, sr = librosa.load(path, sr=16000)
    return y, sr


def amp_to_db(x):
    return 20.0 * torch.log10(torch.max(torch.tensor(1e-5), x))


def normalize(S):
    return torch.clip(S / 100, -1.0, 0.0) + 1.0


def wav2spec(wav):
    if len(wav.shape) == 3:
        wav = wav.squeeze(1)
    D = torch.stft(wav, n_fft=512, return_complex=True).unsqueeze(1)
    S = amp_to_db(torch.abs(D)) - 20
    S, D = normalize(S), torch.angle(D)
    return S, D


def db_to_amp(x):
    return torch.pow(10.0, x * 0.05)


def denormalize(S):
    return (torch.clip(S, 0.0, 1.0) - 1.0) * 100


def istft(mag, phase):
    stft_matrix = mag * torch.exp(1j * phase)
    return torch.istft(stft_matrix, n_fft=512)


def spec2wav(spectrogram, phase):
    S = db_to_amp(denormalize(spectrogram) + 20)
    # spec = torch.cat([spectrogram.unsqueeze(-1), phase.unsqueeze(-1)], dim=-1)
    # spec = torch.view_as_complex(spec)
    return istft(S, phase)


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace