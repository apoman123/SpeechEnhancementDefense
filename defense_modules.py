import random
import math

import torch.nn as nn
import torch
import torchaudio
from torchaudio.functional import add_noise
from torchaudio.transforms import MelSpectrogram
import torch.nn.functional as F

from dns_mpsenet_components.stfts import mag_pha_stft as dns_mpsenet_stft
from dns_mpsenet_components.stfts import mag_pha_istft as dns_mpsenet_istft
from defensegan_components.util_defense_GAN import get_z_star, get_z_sets
from wavepurifier_components import RevGuidedDiffusion, wavepurifier_wav2spec, wavepurifier_spec2wav
from consistency_components.consistency_models.consistency import ConsistencySamplingAndEditing


def get_real_noise(path):
    sample, sr = torchaudio.load(path)
    if sr != 16000:
        sample = torchaudio.functional.resample(sample, sr, 16000)
    return sample

def preprocess_noise(input_wav, noise):
    assert len(input_wav.shape) == 2
    batch_size, length = input_wav.shape
    noise = noise.expand(batch_size, -1)
    
    # too long chop off
    if noise.shape[-1] >= length:
        noise = noise[:, :length]
    else:
        repeat_times = math.ceil(length / noise.shape[-1])
        noise = noise.repeat(1, repeat_times)[:, :length]

    return noise

def preprocess_and_add_noise(input_wav, noise, snr): # b, t
    assert len(input_wav.shape) == 2
    batch_size, length = input_wav.shape
    noise = noise.expand(batch_size, -1)
    
    # too long chop off
    if noise.shape[-1] >= length:
        noise = noise[:, :length]
    else:
        repeat_times = math.ceil(length / noise.shape[-1])
        noise = noise.repeat(1, repeat_times)[:, :length]
        
    # adding
    return torchaudio.functional.add_noise(input_wav, noise, snr=torch.tensor([snr]).to(input_wav.device))

    
class Identity():
    def __init__(self):
        pass

    def __call__(self, x):
        return x


def zero_mean_normalization(data):
    mean = torch.mean(data, dim=-1, keepdim=True).expand(data.shape)
    std = torch.std(data, dim=-1, keepdim=True).expand(data.shape)
    data = (data - mean) / std
    return data

def get_mean_std(data):
    mean = torch.mean(data, dim=-1, keepdim=True).expand(data.shape)
    std = torch.std(data, dim=-1, keepdim=True).expand(data.shape)
    return mean, std
    
    
class Xvector_Features(nn.Module):
    def __init__(self, sample_rate=16000, num_mels=20, window_length=400, zero_mean=False):
        super(Xvector_Features, self).__init__()
        self.zero_mean = zero_mean
        self.mfcc = torchaudio.transforms.MFCC(16000, 60)
        # self.compute_deltas = torchaudio.transforms.ComputeDeltas(window_length)

    def forward(self, x):
        if len(x.shape) == 3:
            x = x.squeeze(1)
            
        if self.zero_mean:
            x = zero_mean_normalization(x)
        mfccs = self.mfcc(x)
        # deltas = self.compute_deltas(mfccs)
        # acceleration = self.compute_deltas(deltas)
        return mfccs
      

        
    
class M5_Features():
    def __init__(self, zero_mean=False):
        self.zero_mean = zero_mean
        
    def __call__(self, x): # b, t
        if self.zero_mean:
            x = zero_mean_normalization(x)
        return x.unsqueeze(1) if len(x.shape) == 2 else x

class M18_Features():
    def __init__(self, zero_mean=False):
        self.zero_mean = zero_mean
        
    def __call__(self, x): # b, t
        if self.zero_mean:
            x = zero_mean_normalization(x)
        return x.unsqueeze(1) if len(x.shape) == 2 else x


class RCNN_Features(nn.Module):
    def __init__(self, zero_mean=False, sample_rate=16000, n_fft=400, win_length=400, hop_length=100, n_mels=40):
        super(RCNN_Features, self).__init__()
        self.zero_mean = zero_mean
        self.transform = MelSpectrogram(
                        sample_rate=sample_rate,
                        n_fft=n_fft,
                        win_length=win_length,
                        hop_length=hop_length,
                        n_mels=n_mels
                    )
        
    def forward(self, x):
        if self.zero_mean:
            x = zero_mean_normalization(x)
        mels = self.transform(x)
        return mels
        

class Unsqueeze():
    def __init__(self):
        pass

    def __call__(self, x):
        return x.unsqueeze(1)

class Squeeze():
    def __init__(self):
        pass

    def __call__(self, x):
        return x.squeeze(1)


class BackgroundNoiseAdding():
    def __init__(self, noise_list, snr=0):
        self.noise_list = noise_list
        self.snr = snr
    
    def __call__(self, x):
        noise = random.choice(self.noise_list).to(x.device)
        return preprocess_and_add_noise(x, noise, snr=self.snr)

class VPSDEBackgroundNoiseAdding():
    def __init__(self, noise_list, beta_min, beta_max, N, total_noise_levels):
        betas = torch.linspace(beta_min/N, beta_max/N, N)
        self.total_noise_levels = total_noise_levels
        self.a = (1 - betas).cumprod(dim=0)
        self.noise_list = noise_list
        
    def __call__(self, wav):
        noise = random.choice(self.noise_list)
        noise = preprocess_noise(wav, noise).to(wav.device)
        noisy_wav = wav * self.a[self.total_noise_levels - 1].sqrt() + noise * (1.0 - self.a[self.total_noise_levels - 1]).sqrt()
        return noisy_wav
        

# noise adding method reference from revised ddpm (VPSDE)
class VPSDENoiseAdding(nn.Module):
    def __init__(self, beta_min, beta_max, N, total_noise_levels):
        super(VPSDENoiseAdding, self).__init__()
        betas = torch.linspace(beta_min/N, beta_max/N, N)
        self.total_noise_levels = total_noise_levels
        self.a = (1 - betas).cumprod(dim=0)
    def forward(self, wav):
        noise = torch.randn_like(wav)
        noisy_wav = wav * self.a[self.total_noise_levels - 1].sqrt() + noise * (1.0 - self.a[self.total_noise_levels - 1]).sqrt()
        return noisy_wav


# one step noise adding
class GaussianNoiseAdding(nn.Module):
    def __init__(self, snr):
        super(GaussianNoiseAdding, self).__init__()
        self.snr = snr
        
    def forward(self, wav): # b, t
        noise = torch.randn_like(wav)
        noisy_wav = add_noise(wav, noise, snr=torch.tensor([self.snr]).to(wav.device))
        return noisy_wav

class GaussianNoiseAddingDBFS(nn.Module):
    def __init__(self, dbfs):
        super(GaussianNoiseAddingDBFS, self).__init__()
        self.target_dbfs = dbfs

    def forward(self, wav):
        noise = self.generate_gaussian_noise_dbfs(wav)
        return wav + noise

    def generate_gaussian_noise_dbfs(self, waveform):
        """
        產生指定 dBFS 響度的 Gaussian noise。
        回傳：
            noise: Tensor，形狀 [channels, length]，符合指定 dBFS
        """
        # Step 1: 產生標準常態分佈的 noise
        noise = torch.randn_like(waveform)
    
        # Step 2: 計算目前 RMS
        rms = torch.sqrt(torch.mean(noise ** 2, dim=1, keepdim=True))  # [batch_size, 1]
    
        # Step 3: 根據目標 dBFS 計算目標 RMS（Full scale = 1.0）
        target_rms = 10 ** (self.target_dbfs / 20)  # scalar
    
        # Step 4: 對每個 channel 正規化
        noise = noise * (target_rms / (rms + 1e-10))  # broadcast RMS 到 [batch_size, length]
    
        return noise

class GaussianNoiseAddingZeroMean(nn.Module):
    def __init__(self, gain):
        super(GaussianNoiseAddingZeroMean, self).__init__()
        self.gain = gain

    def forward(self, wav):
        wav = zero_mean_normalization(wav)
        noise = zero_mean_normalization(torch.randn_like(wav)) * self.gain
        return zero_mean_normalization(wav + noise)


class SimpleDefenseModule():
    def __init__(self, method):
        self.method = method

    def __call__(self, wav):
        return self.method(wav)

class DefenseGANDefenseModule(nn.Module):
    def __init__(self, defense_gan, lr=10, rec_iter=10, rec_rr=2):
        super(DefenseGANDefenseModule, self).__init__()
        self.defense_gan = defense_gan
        self.lr = lr
        self.rec_iter = 200
        self.rec_rr = 10
        self.loss = nn.L1Loss()
        
    def forward(self, wav):
        wavs = torch.split(wav, 65536, dim=-1)
        reconstructed_wavs = []
        for wav in wavs:
            # padding
            b, c, t = wav.shape
            wav = torch.cat([
                wav,
                torch.zeros(b, c, 65536 - t).to(wav.device)
            ], dim=-1)
            
            # get z 
            _, z_sets = get_z_sets(self.defense_gan, wav, self.lr, 
                                   self.loss, wav.device, rec_iter = self.rec_iter, 
                                   rec_rr = self.rec_rr, input_latent = 100, global_step = 3.0)
    
            z_star = get_z_star(self.defense_gan, wav, z_sets, self.loss, wav.device).to(wav.device)
    
            # generate data
            recons_wav = self.defense_gan(z_star)[:, :, :t]
            reconstructed_wavs.append(recons_wav)
        
        return torch.cat(reconstructed_wavs, dim=-1)
        
class AudioPureDefenseModule(nn.Module):
    def __init__(self, audio_pure):
        super(AudioPureDefenseModule, self).__init__()
        self.audio_pure = audio_pure

    def forward(self, wav):
        if len(wav.shape) == 3:
            wav = wav.squeeze(1)
        self.audio_pure.rev_vpsde.audio_shape = (1, wav.shape[-1])
        return self.audio_pure(wav.unsqueeze(1)).squeeze(1)

class WavePurifierDefenseModule(nn.Module):
    def __init__(self, args, config):
        super(WavePurifierDefenseModule, self).__init__()
        self.diffmodel = RevGuidedDiffusion(args, config, device=config.device)

    def forward(self, wav):
        if len(wav.shape) == 2:
            wav = wav.unsqueeze(1)
        
        # get spectrogram
        S, P = wavepurifier_wav2spec(wav)
        nframes = S.shape[3]
        nwindow = nframes  // 256
        
        # purifying
        purified_specs = []
        batch_size, channel, mels, time = S.shape
        if nwindow == 0:    # The to be purified spec is **shorter** than the window
            # Padding the S
            offset = 256-nframes  # pad start part
            pad = torch.zeros(batch_size, channel, 257, offset).to(S.device)
            S = torch.cat((S, pad), dim=-1)
            # su = S[:, :, :256, :].contiguous()
            # su = torch.cat((S[:, :, :256, :], S[:, :, :256, :], S[:, :, :256, :]), axis=0).reshape(batch_size, 3, 256, 256).contiguous()  # Stack 3 layers
            su = S[:, :, :256, :]
            counter = 0
            # AE_name = AE_path.split("/")[-1][:-4]
            # tag = str(args.t) + "_" + AE_name
            # Purify the only window
            purified_spec = self.diffmodel.image_editing_sample((su - 0.5) * 2, 0, bs_id=counter)
            purified_specs.append(purified_spec)
        else:
            restlength = nframes - nwindow * 256  # 163
            offset = 256 - restlength  # 93
            # If the length is longer than 256, then we build a list to purify them one by one
            sw = []  # record window of frames
            for w in range(nwindow):
                # Those are the complete window
                sw.append(S[:, :, :256, w * 256:w * 256 + 256])
            sw.append(S[:, :, :256, -256:])  # This is the last window
            # AE_name = AE_path.split("/")[-1][:-4]
            window_id = 0
            for kwin in range(len(sw)):
                # Purify every window
                # su = torch.cat((sw[kwin], sw[kwin], sw[kwin]), axis=0).reshape(batch_size, 3, 256, 256).contiguous()  # Stack 3 layers
                su = sw[kwin]
                # su = sw[kwin].contiguous()
                # su = torch.Tensor(su).contiguous()
                counter = 0
                # tag = str(args.t_s) + "_" +str(args.t_m) + "_"+str(args.t_l) + "_" + AE_name
                # this step save the specs to ./logs/[tag]
                purified_spec = self.diffmodel.image_editing_sample((su - 0.5) * 2, window_id, bs_id=counter)
                window_id = window_id + 1
                purified_specs.append(purified_spec)

        specs = torch.cat(purified_specs, dim=-1)
        _, _, _, real_time_len = P.shape
        _, _, _, time_len = specs.shape
        specs = specs[:, 0, :, :real_time_len].unsqueeze(1)
        su = torch.cat([specs, S[:, :, 256, :real_time_len].unsqueeze(2)], dim=-2)
        purified_wav = wavepurifier_spec2wav(su.squeeze(1), P.squeeze(1))
        return purified_wav


class LogMelSpectrogram(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.melspctrogram = MelSpectrogram(
            sample_rate=16000,
            n_fft=1024,
            win_length=1024,
            hop_length=160,
            center=False,
            power=1.0,
            norm="slaney",
            onesided=True,
            n_mels=128,
            mel_scale="slaney",
        )

    def forward(self, wav):
        wav = F.pad(wav, ((1024 - 160) // 2, (1024 - 160) // 2), "reflect")
        mel = self.melspctrogram(wav)
        logmel = torch.log(torch.clamp(mel, min=1e-5))
        return logmel

class ConsistencyModelDefenseModule(nn.Module):
    def __init__(self, consistency_model):
        super(ConsistencyModelDefenseModule, self).__init__()
        # feature
        self.transform = LogMelSpectrogram()

        # defense modules
        self.consistency_model = consistency_model
        self.sigmas = [0.25, 0.5, 1]
        self.hifigan = torch.hub.load("bshall/hifigan:main", "hifigan_hubert_soft")
        self.consistency_sampling_and_editing = ConsistencySamplingAndEditing(
                                                    sigma_min = 0.25, # minimum std of noise
                                                    sigma_data = 0.5, # std of the data
                                                )
        
    def forward(self, wavs):
        # feature extraction
        specs = self.transform(wavs)
        spec_list = torch.split(specs, 128, dim=-1)

        purified_results = []
        for spec in spec_list:
            b, c, mels, t = spec.shape
            # padding
            if spec.shape[-1] < 128:
                spec = torch.cat(
                        [
                            spec,
                            torch.zeros(b, c, mels, 128-t).to(spec.device)
                        ],
                        dim=-1
                    )
            spec = self.consistency_sampling_and_editing(
                        self.consistency_model, # student model or any trained model
                        spec, # used to infer the shapes
                        sigmas=[random.choice(self.sigmas)], # sampling starts at the maximum std (T)
                        clip_denoised=False, # whether to clamp values to [-1, 1] range
                        verbose=False,
                        start_from_y=True
                    )
        
            purified_results.append(spec[:, :, :, :t])
            
        final_spec = torch.cat(purified_results, dim=-1).squeeze(1)
        purified_wavs = self.hifigan(final_spec)
        return purified_wavs


class DNSMPSENetDefenseModule(nn.Module):
    def __init__(self, mpsenet, noise_adding, n_fft, hop_size, win_size, compress_factor):
        super(DNSMPSENetDefenseModule, self).__init__()
        self.noise_adding = noise_adding
        self.mpsenet = mpsenet
        self.n_fft = n_fft
        self.hop_size = hop_size
        self.win_size = win_size
        self.compress_factor = compress_factor
        
    def forward(self, noisy_wav, add_noise=True):
        if len(noisy_wav.shape) == 3:
            noisy_wav = noisy_wav.squeeze(1)
        # add noise
        if add_noise:
            noisy_wav = self.noise_adding(noisy_wav)

        # denoise
        norm_factor = torch.sqrt(len(noisy_wav) / torch.sum(noisy_wav ** 2.0)).to(noisy_wav.device)
        noisy_audio = (noisy_wav * norm_factor)
        noisy_amp, noisy_pha, noisy_com = dns_mpsenet_stft(noisy_audio, self.n_fft, self.hop_size, self.win_size, self.compress_factor)
        amp_g, pha_g, com_g = self.mpsenet(noisy_amp, noisy_pha)
        audio_g = dns_mpsenet_istft(amp_g, pha_g, self.n_fft, self.hop_size, self.win_size, self.compress_factor)
        audio_g = audio_g / norm_factor

        return audio_g


class Classifier(nn.Module):
    def __init__(self, classifier, feature_extraction):
        super(Classifier, self).__init__()
        self.classifier = classifier
        self.feature_extraction = feature_extraction

    def get_embeddings(self, wav):
        features = self.feature_extraction(wav)
        return self.classifier.get_embeddings(features)
        
    def forward(self, wav):
        features = self.feature_extraction(wav)
        result = self.classifier(features)
        return result
    

class AcousticSystem(nn.Module):
    def __init__(self, defense_module, classifier):
        super(AcousticSystem, self).__init__()
        self.defense_module = defense_module
        self.classifier = classifier
    
    def forward(self, wav):
        wav = self.defense_module(wav)
        classification_result = self.classifier(wav)
        return classification_result
        
        

