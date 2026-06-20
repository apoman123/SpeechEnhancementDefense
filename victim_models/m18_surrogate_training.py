import time
import argparse
import os
import json
import sys
sys.path.append('/home/apoman123/speech_enhancement_adversarial_evaluation')

import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from datasets import Audio, load_from_disk, concatenate_datasets
from tqdm import tqdm
from m18 import M18
from dns_mpsenet_components.generator import MPNet

from defense_modules import Classifier, AcousticSystem

class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

def zero_mean_normalization(data):
    mean = torch.mean(data, dim=-1, keepdim=True).expand(data.shape)
    std = torch.std(data, dim=-1, keepdim=True).expand(data.shape)
    data = (data - mean) / std
    return data

def soft_cross_entropy(pred_logits, soft_targets, temperature=1):
    soft_targets = F.softmax(soft_targets / temperature, dim=-1)
    soft_prob = F.log_softmax(pred_logits / temperature, dim=-1)
    soft_targets_loss = torch.sum(soft_targets * (soft_targets.log() - soft_prob)) / soft_prob.size()[0] * (temperature**2)
    return soft_targets_loss

def hard_label_loss(student_logits, teacher_logits):
    teacher_label = torch.argmax(F.softmax(teacher_logits, dim=-1), dim=-1)
    loss = F.nll_loss(F.log_softmax(student_logits, dim=-1), teacher_label)
    return loss


def load_config(config_path):
    """Load configuration from a YAML file."""
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def get_noise_module(args):
    if args.noise_adding_method == "VPSDE":
        from defense_modules import VPSDENoiseAdding
        return VPSDENoiseAdding(args.beta_min, args.beta_max, args.N, args.noise_steps)
    elif args.noise_adding_method == "Gaussian":
        from defense_modules import GaussianNoiseAdding
        return GaussianNoiseAdding(args.noise_snr)
    elif args.noise_adding_method == "Background":
        from defense_modules import BackgroundNoiseAdding
        esc50_filtered = load_from_disk(args.noise_path).cast_column("audio", Audio(sampling_rate=16000))
        noise_list = [torch.from_numpy(data['audio']['array']).float() for data in esc50_filtered]
        return BackgroundNoiseAdding(noise_list, args.snr)
    elif args.noise_adding_method == "VPSDEBackground":
        from defense_modules import VPSDEBackgroundNoiseAdding
        esc50_filtered = load_from_disk(args.noise_path).cast_column("audio", Audio(sampling_rate=16000))
        noise_list = [torch.from_numpy(data['audio']['array']).float() for data in esc50_filtered]
        return VPSDEBackgroundNoiseAdding(noise_list, args.beta_min, args.beta_max, args.N, args.noise_steps)
    elif args.noise_adding_method == "GaussianDBFS":
        from defense_modules import GaussianNoiseAddingDBFS
        return GaussianNoiseAddingDBFS(args.noise_dbfs)
    elif args.noise_adding_method == "GaussianZeroMean":
        from defense_modules import GaussianNoiseAddingZeroMean
        return GaussianNoiseAddingZeroMean(args.noise_gain)
    else:
        return Identity()

def get_classifier(args):
    if args.dataset == "sc09":
        from victim_models import M18
        from defense_modules import M18_Features
        victim = M18(n_input=1, n_output=10)
        transform = M18_Features(zero_mean=args.zero_mean)
        victim.load_state_dict(torch.load(args.classifier_path, weights_only=False))
        
    elif args.dataset == "vctk":
        from victim_models import X_vector
        from defense_modules import Xvector_Features
        victim = X_vector(60, 108)
        transform = Xvector_Features(zero_mean=args.zero_mean)
        classifier_weights = {}
        with safe_open(args.classifier_path, framework="pt") as f:
            for key in f.keys():
                classifier_weights[key] = f.get_tensor(key)
        
        victim.load_state_dict(classifier_weights)

    elif args.dataset == "qkws":
        from victim_models.RCNN import KWSModel
        from defense_modules import RCNN_Features
        victim = KWSModel()
        transform = RCNN_Features(zero_mean=args.zero_mean)
        classifier_weights = {}
        with safe_open(args.classifier_path, framework="pt") as f:
            for key in f.keys():
                classifier_weights[key] = f.get_tensor(key)
        
        victim.load_state_dict(classifier_weights)
    classifier = Classifier(victim, transform)
    return classifier


def get_defense_module(args):
    noise_adding = get_noise_module(args)
    if args.defense_module == "SEMamba":
        from defense_modules import SEMambaDefenseModule
        cfg = load_config(args.config_path)
        model = SEMamba(cfg)
        state_dict = torch.load(args.defense_model_path, map_location="cpu")
        model.load_state_dict(state_dict['generator'])
        return SEMambaDefenseModule(model, noise_adding, args.n_fft, args.hop_size, args.win_size, args.compress_factor)

    elif args.defense_module == "MambaSEUNet":
        from defense_modules import SEMambaDefenseModule
        cfg = load_config(args.config_path)
        model = MambaSEUNet(cfg)
        state_dict = torch.load(args.defense_model_path, map_location="cpu")
        model.load_state_dict(state_dict['generator'])
        return SEMambaDefenseModule(model, noise_adding, args.n_fft, args.hop_size, args.win_size, args.compress_factor)

    elif args.defense_module == "MPSENet":
        from defense_modules import DNSMPSENetDefenseModule
        config_file = os.path.join(args.config_path)
        with open(config_file) as f:
            data = f.read()
        json_config = json.loads(data)
        h = AttrDict(json_config)
        
        model = MPNet(h)
        state_dict = torch.load(args.defense_model_path, map_location="cpu")
        model.load_state_dict(state_dict['generator'])
        return DNSMPSENetDefenseModule(model, noise_adding, args.n_fft, args.hop_size, args.win_size, args.compress_factor)
        
    elif args.defense_module == "ZipEnhancer":
        from defense_modules import ZipEnhancerDefenseModule
        kwargs = Config.from_file(args.config_path)['model']
        h = dict(
            num_tsconformers=kwargs['num_tsconformers'],
            dense_channel=kwargs['dense_channel'],
            former_conf=kwargs['former_conf'],
            batch_first=kwargs['batch_first'],
            model_num_spks=kwargs['model_num_spks'],
        )
        h = AttrDict(h)
        model = ZipEnhancer(h)
        model.load_state_dict(torch.load(model_file, map_location='cpu')['generator'])
        return ZipEnhancerDefenseModule(model, noise_adding, args.n_fft, args.hop_size, args.win_size)

    elif args.defense_module == "CMGAN":
        from defense_modules import CMGANDefenseModule
        cmgan = TSCNet(num_channel=64, num_features=n_fft // 2 + 1)
        cmgan.load_state_dict(torch.load("/home/apoman123/CMGAN/src/best_ckpt/ckpt", map_location="cpu"))
        return CMGANDefenseModule(cmgan, noise_module, 400, 100)

    elif args.defense_module == "AudioPure":
        from diffusion_models.diffwave_sde import RevDiffWave
        from defense_modules import AudioPureDefenseModule
        audio_pure = RevDiffWave(args)
        return AudioPureDefenseModule(audio_pure)
    elif args.defense_module in ["AS", "MS", "AT"]:
        from defense_modules import SimpleDefenseModule
        method = TimeDomainDefense(args.defense_module)
        return SimpleDefenseModule(method)
    elif args.defense_module in ["DS", "LPF", "BPF"]:
        from defense_modules import SimpleDefenseModule
        method = FreqDomainDefense(args.defense_module)
        return SimpleDefenseModule(method)
    elif args.defense_module == "Identity":
        return Identity()

def test(model, data_loader, verbose=False):
    global layer_norm
    """Measures the accuracy of a model on a data set.""" 
    # Make sure the model is in evaluation mode.
    model.eval()
    correct = 0
#     print('----- Model Evaluation -----')
    # We do not need to maintain intermediate activations while testing.
    with torch.no_grad():
        # Loop over test data.
        for features, target in tqdm(data_loader, total=len(data_loader.batch_sampler), desc="Testing"):
            # Forward pass.
            features = zero_mean_normalization(features)
            output = model(features.to(device))
            # Get the label corresponding to the highest predicted probability.
            pred = output.argmax(dim=1, keepdim=True)
            # Count number of correct predictions.
            correct += pred.cpu().eq(target.view_as(pred)).sum().item()
    # Print test accuracy.
    percent = 100. * correct / len(data_loader.sampler)
    if verbose:
        print(f'Test accuracy: {correct} / {len(data_loader.sampler)} ({percent:.0f}%)')
    return percent


def train(model, surrogate_model, data_loader, test_loader, optimizer, num_epochs, args):
    # global layer_norm
    """Simple training loop for a PyTorch model.""" 
    
    # Move model to the device (CPU or GPU).
    model.to(device)
    surrogate_model.to(device)
    
    accs = []
    # Exponential moving average of the loss.
    ema_loss = None

#     print('----- Training Loop -----')
    # Loop over epochs.
    for epoch in range(num_epochs):
        tick = time.time()
        model.eval()
        surrogate_model.train()
        # Loop over data.
        for batch_idx, (features, target) in tqdm(enumerate(data_loader), total=len(data_loader.batch_sampler), desc="training"):
            # Forward pass.
            features = zero_mean_normalization(features)
            student_distribution = surrogate_model(features.to(device))
            with torch.no_grad():
                teacher_distribution = model(features.to(device))

            # loss = criterion(output.to(device), target.to(device))

            # sof6t label
            # soft_label_loss = soft_cross_entropy(student_distribution, teacher_distribution)
            # ce_loss = F.cross_entropy(student_distribution, target)
            # loss = 0.75 * ce_loss + 0.25 soft_label_loss

            # hard label
            loss = hard_label_loss(student_distribution, teacher_distribution)
            # loss = soft_label_loss
            loss.backward()
            
            if (batch_idx+1) % 5 == 0:
                # Backward pass.
                optimizer.step()
                optimizer.zero_grad()
            # NOTE: It is important to call .item() on the loss before summing.
            if ema_loss is None:
                ema_loss = loss.item()
            else:
                ema_loss += (loss.item() - ema_loss) * 0.01
            tock = time.time()
        acc = test(surrogate_model, test_loader, verbose=True)
        accs.append(acc)
        # Print out progress the end of epoch.
        print('Epoch: {} \tLoss: {:.6f} \t Time taken: {:.6f} seconds'.format(epoch+1, ema_loss, tock-tick),)
        torch.save(surrogate_model.state_dict(), os.path.join(args.checkpoint_path, f"m18net_{epoch}.pt"))
        print("Model Saved!")
        # if os.path.isfile(f'model_{epoch-1}.ckpt'):
        #     os.remove(f'model_{epoch-1}.ckpt')
    return accs


# using glorot initialization
def init_weights(m):
    if isinstance(m, torch.nn.Conv1d):
        torch.nn.init.xavier_uniform_(m.weight.data)
class ToMono(torch.nn.Module):
    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        return torch.mean(waveform, dim=0, keepdim=True)

class Normalize(torch.nn.Module):
    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        return (waveform-waveform.mean()) / waveform.std()

class Pad(torch.nn.Module):
    def __init__(self, value: float, size: int):
        super(Pad, self).__init__()
        self.value = value
        self.size = size
    
    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.pad(waveform, (0, self.size-max(waveform.shape)), "constant", self.value)
        
def collate_fn(batch):
    lengths = [data['audio']['array'].shape[-1] for data in batch]
    max_length = max(lengths)
    num_pads = [max_length - length for length in lengths]
    tensors = torch.stack(
        [
            torch.cat([
            torch.from_numpy(data['audio']['array']).float(), torch.zeros(pads)
            ], dim=0)
            for data, pads in zip(batch, num_pads)
        ]
    )
    # tensors = zero_mean_normalization(tensors)
    labels = torch.tensor([data['label'] for data in batch])
    return tensors.unsqueeze(1), labels
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_path", default="", type=str)
    parser.add_argument("--log_path", default="", type=str)
    parser.add_argument("--dataset_path")
    parser.add_argument("--classifier_path")
    parser.add_argument("--noise_adding_method", default="VPSDE", choices=["VPSDE", "Gaussian", "Identity", "VPSDEBackground", "GaussianDBFS", "GaussianZeroMean"])
    parser.add_argument("--noise_dbfs", default=-20, type=int)
    parser.add_argument("--defense_model_path")
    parser.add_argument("--config_path")
    parser.add_argument("--dataset", default="sc09")
    parser.add_argument("--zero_mean", default=True)
    parser.add_argument("--defense_module")
    parser.add_argument("--n_fft", default=400, type=int)
    parser.add_argument("--hop_size", default=100, type=int)
    parser.add_argument("--win_size", default=400, type=int)
    parser.add_argument("--compress_factor", default=0.3, type=float)
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    speech_commands = load_from_disk(args.dataset_path).cast_column("audio", Audio(sampling_rate=16000))
    # augmented_training_set = load_from_disk("/data/nas07/PersonalData/apoman123/sc09_augmented_training_set")
    
    # training_set = concatenate_datasets([speech_commands['train'], augmented_training_set])

    batch_size = 8
    train_loader = torch.utils.data.DataLoader(
        speech_commands["train"],
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=10
    )
    test_loader = torch.utils.data.DataLoader(
        speech_commands["validation"],
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        collate_fn=collate_fn,
    )

    classifier = get_classifier(args)
    defense_module = get_defense_module(args)
    model = AcousticSystem(defense_module, classifier)
    
    surrogate_model = M18(
        n_input=1,
        n_output=10
    )

    
    # global layer_norm
    # layer_norm = nn.LayerNorm(1, elementwise_affine=False)
    # print("Num Parameters:", sum([p.numel() for p in model.parameters()]))

    # audio_transform = torch.nn.Sequential(*[
    #     Normalize() # normalize audio signal to have mean=0 & std=1
    # ])
    
    # criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(surrogate_model.parameters(), weight_decay=1e-4) #L2 regularization is added
    writer = SummaryWriter(log_dir=args.log_path)
    num_epochs = 100
    accs = train(model, surrogate_model, train_loader, test_loader, optimizer, num_epochs=num_epochs, args=args)
    for idx in range(len(accs)):
        writer.add_scalar("Test Acc", accs[idx], idx)
    
    