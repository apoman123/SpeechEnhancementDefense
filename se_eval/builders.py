"""Factory helpers shared by every evaluation entry point.

These functions used to be copy-pasted (with small, drifting differences) into
each ``*_evaluation.py`` script.  They now live here as the single source of
truth.  Heavy / optional model dependencies are imported lazily inside the
branch that needs them, so importing this module stays cheap.
"""
import argparse
import json
import os

import torch
from datasets import Audio, load_from_disk
from safetensors import safe_open

from attacks.fakebob import FakeBob
from attacks.pgd import AudioAttack
from defense_modules import AcousticSystem, Classifier, Identity
from simple_defenses import FreqDomainDefense, TimeDomainDefense

from . import config


class AttrDict(dict):
    """Dictionary that also exposes its keys as attributes."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__dict__ = self


def dict2namespace(cfg):
    """Recursively convert a nested dict into an ``argparse.Namespace``."""
    namespace = argparse.Namespace()
    for key, value in cfg.items():
        setattr(namespace, key, dict2namespace(value) if isinstance(value, dict) else value)
    return namespace


def load_config(config_path):
    """Load a YAML configuration file."""
    import yaml

    with open(config_path, "r") as file:
        return yaml.safe_load(file)


def short_audio(batch):
    """Filter predicate keeping clips shorter than 5 seconds."""
    return batch["audio"]["array"].shape[-1] < config.SAMPLE_RATE * 5


# ---------------------------------------------------------------------------
# Noise injection (the randomised pre-processing applied before enhancement)
# ---------------------------------------------------------------------------
def get_noise_module(args):
    method = args.noise_adding_method
    if method == "VPSDE":
        from defense_modules import VPSDENoiseAdding

        return VPSDENoiseAdding(args.beta_min, args.beta_max, args.N, args.noise_steps)
    if method == "Gaussian":
        from defense_modules import GaussianNoiseAdding

        return GaussianNoiseAdding(args.noise_snr)
    if method == "GaussianDBFS":
        from defense_modules import GaussianNoiseAddingDBFS

        return GaussianNoiseAddingDBFS(args.noise_dbfs)
    if method == "GaussianZeroMean":
        from defense_modules import GaussianNoiseAddingZeroMean

        return GaussianNoiseAddingZeroMean(args.noise_gain)
    if method == "Background":
        from defense_modules import BackgroundNoiseAdding

        noise_list = _load_noise_list(args.noise_path)
        return BackgroundNoiseAdding(noise_list, args.snr)
    if method == "VPSDEBackground":
        from defense_modules import VPSDEBackgroundNoiseAdding

        noise_list = _load_noise_list(args.noise_path)
        return VPSDEBackgroundNoiseAdding(
            noise_list, args.beta_min, args.beta_max, args.N, args.noise_steps
        )
    return Identity()


def _load_noise_list(noise_path):
    noise_ds = load_from_disk(noise_path).cast_column(
        "audio", Audio(sampling_rate=config.SAMPLE_RATE)
    )
    return [torch.from_numpy(item["audio"]["array"]).float() for item in noise_ds]


# ---------------------------------------------------------------------------
# Defense modules (speech enhancement / purification front-ends)
# ---------------------------------------------------------------------------
def get_defense_module(args):
    noise_adding = get_noise_module(args)
    name = args.defense_module

    if name == "MPSENet":
        from dns_mpsenet_components.generator import MPNet
        from defense_modules import DNSMPSENetDefenseModule

        with open(args.config_path) as f:
            h = AttrDict(json.loads(f.read()))
        model = MPNet(h)
        model.load_state_dict(torch.load(args.defense_model_path, map_location="cpu")["generator"])
        return DNSMPSENetDefenseModule(
            model, noise_adding, args.n_fft, args.hop_size, args.win_size, args.compress_factor
        )

    if name == "AudioPure":
        from diffusion_models.diffwave_sde import RevDiffWave
        from defense_modules import AudioPureDefenseModule

        return AudioPureDefenseModule(RevDiffWave(args))

    if name == "Consistency":
        from consistency_components.unet import UNetModel
        from defense_modules import ConsistencyModelDefenseModule

        consistency = UNetModel(
            image_size=128,
            in_channels=1,
            model_channels=192,
            out_channels=1,
            num_res_blocks=3,
            attention_resolutions=tuple("32,16,8"),
            dropout=0.0,
            channel_mult=(1, 1, 2, 3, 4),
            num_classes=None,
            use_checkpoint=False,
            use_fp16=False,
            num_heads=4,
            num_head_channels=64,
            num_heads_upsample=-1,
            use_scale_shift_norm=True,
            resblock_updown=True,
            use_new_attention_order=False,
        )
        consistency.load_state_dict(torch.load(args.consistency_path))
        return ConsistencyModelDefenseModule(consistency)

    if name == "DefenseGAN":
        from defensegan_components.wavegan import WaveGANGenerator
        from defense_modules import DefenseGANDefenseModule

        generator = WaveGANGenerator(
            num_channels=1, model_size=32, use_batch_norm=False, slice_len=65536
        )
        generator.load_state_dict(torch.load(args.defense_gan_path)["generator"])
        return DefenseGANDefenseModule(generator)

    if name in ("AS", "MS", "AT"):
        from defense_modules import SimpleDefenseModule

        return SimpleDefenseModule(TimeDomainDefense(name))

    if name in ("DS", "LPF", "BPF"):
        from defense_modules import SimpleDefenseModule

        return SimpleDefenseModule(FreqDomainDefense(name))

    if name == "Identity":
        return Identity()

    raise ValueError(
        f"Unsupported defense_module '{name}'. Supported values: {sorted(DEFENSE_MODULES)}."
    )


#: Defense front-ends wired up in :func:`get_defense_module`.
DEFENSE_MODULES = (
    "MPSENet",
    "AudioPure",
    "Consistency",
    "DefenseGAN",
    "Identity",
    "AS",
    "MS",
    "AT",
    "DS",
    "LPF",
    "BPF",
)

NOISE_METHODS = (
    "VPSDE",
    "Gaussian",
    "GaussianDBFS",
    "GaussianZeroMean",
    "Background",
    "VPSDEBackground",
    "Identity",
)


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------
def get_dataset(args):
    path = config.DATASET_PATHS[args.dataset]
    ds = load_from_disk(path).cast_column("audio", Audio(sampling_rate=config.SAMPLE_RATE))

    if args.dataset == "vctk":
        ds = ds.shuffle(seed=42).train_test_split(test_size=0.05, shuffle=False)
        ds = ds["test"].filter(short_audio)
    else:
        ds = ds["test"]

    limit = getattr(args, "limit", None)
    if limit:
        ds = ds.select(range(min(limit, len(ds))))
    return ds


# ---------------------------------------------------------------------------
# Victim classifiers
# ---------------------------------------------------------------------------
def build_classifier(dataset, classifier_path, zero_mean=False):
    """Construct the victim classifier + feature extractor for ``dataset``."""
    if dataset == "sc09":
        from victim_models import M18
        from defense_modules import M18_Features

        victim = M18(n_input=1, n_output=10)
        transform = M18_Features(zero_mean=zero_mean)
        victim.load_state_dict(torch.load(classifier_path, weights_only=False))

    elif dataset == "vctk":
        from victim_models import X_vector
        from defense_modules import Xvector_Features

        victim = X_vector(60, 108)
        transform = Xvector_Features(zero_mean=zero_mean)
        victim.load_state_dict(_load_safetensors(classifier_path))

    elif dataset == "qkws":
        from victim_models.RCNN import KWSModel
        from defense_modules import RCNN_Features

        victim = KWSModel()
        transform = RCNN_Features(zero_mean=zero_mean)
        victim.load_state_dict(_load_safetensors(classifier_path))

    else:
        raise ValueError(f"Unknown dataset '{dataset}'.")

    return Classifier(victim, transform)


def _load_safetensors(path):
    weights = {}
    with safe_open(path, framework="pt") as f:
        for key in f.keys():
            weights[key] = f.get_tensor(key)
    return weights


def get_classifier(args):
    return build_classifier(args.dataset, args.classifier_path, args.zero_mean)


def get_surrogate_classifier(args):
    return build_classifier(
        args.dataset, args.surrogate_classifier_path, args.surrogate_zero_mean
    )


def build_acoustic_system(args):
    """Build the full ``defense -> classifier`` pipeline used by the attacks."""
    return AcousticSystem(get_defense_module(args), get_classifier(args))


# ---------------------------------------------------------------------------
# Attacks
# ---------------------------------------------------------------------------
def get_attack(args, model, steps=None, eot_size=None):
    """Instantiate an attack against ``model``.

    ``steps`` / ``eot_size`` override the corresponding CLI arguments, which lets
    a caller sweep over those values without rebuilding ``args``.
    """
    steps = args.steps if steps is None else steps
    eot_size = args.eot_size if eot_size is None else eot_size

    if args.attack_type == "pgd":
        learning_rate = args.epsilon / 5 if args.bound_norm == "linf" else args.epsilon / 2
        return AudioAttack(
            model=model,
            eps=args.epsilon,
            norm=args.bound_norm,
            max_iter_1=steps,
            max_iter_2=0,
            learning_rate_1=learning_rate,
            eot_attack_size=eot_size,
            eot_defense_size=1,
            verbose=False,
        )

    if args.attack_type == "fakebob":
        task = "CSI" if args.dataset == "vctk" else "SCR"
        return FakeBob(
            model=model,
            task=task,
            targeted=False,
            verbose=0,
            confidence=0.5,
            epsilon=args.epsilon,
            max_lr=0.001,
            min_lr=1e-6,
            max_iter=200,
            samples_per_draw=200,
            samples_per_draw_batch_size=8,
            batch_size=args.batch_size,
        )

    if args.attack_type == "torchattacks_pgd":
        from attacks.torchattacks_pgd import PGD

        return PGD(model=model, eps=args.epsilon, steps=steps, alpha=args.alpha)

    raise ValueError(f"Unknown attack_type '{args.attack_type}'.")


# ---------------------------------------------------------------------------
# Batching
# ---------------------------------------------------------------------------
def collate(samples):
    """Pad a list of variable-length clips into a ``(B, 1, T)`` batch."""
    input_values = [torch.from_numpy(s["audio"]["array"]).float() for s in samples]
    max_len = max(v.shape[-1] for v in input_values)
    input_values = [
        torch.cat([v, torch.zeros(max_len - v.shape[-1])], dim=-1) for v in input_values
    ]
    labels = torch.tensor([s["label"] for s in samples])
    input_values = torch.stack(input_values, dim=0)
    return {"input_values": input_values.unsqueeze(1), "labels": labels}
