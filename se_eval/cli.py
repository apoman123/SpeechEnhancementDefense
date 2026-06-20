"""Shared command-line argument definitions for the evaluation scripts."""
import argparse

from . import config
from .builders import DEFENSE_MODULES, NOISE_METHODS

ATTACK_TYPES = ("pgd", "pgd+eot", "fakebob", "torchattacks_pgd")


def build_parser(description=None):
    """Return an :class:`argparse.ArgumentParser` with every shared argument.

    Individual entry points add their own extra flags on top of this parser.
    """
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--backend", default="nccl", type=str)

    # dataset configuration
    parser.add_argument("--dataset", default="vctk", choices=["vctk", "sc09", "qkws"])
    parser.add_argument("--num_workers", default=4, type=int)
    parser.add_argument("--batch_size", default=512, type=int)
    parser.add_argument("--pin_memory", default=True, type=bool)
    parser.add_argument(
        "--limit", default=None, type=int,
        help="Evaluate on at most this many test clips (default: whole test split).",
    )

    # defense configuration
    parser.add_argument("--defense_module", default="Identity", choices=list(DEFENSE_MODULES))
    parser.add_argument("--noise_adding_method", default="Identity", choices=list(NOISE_METHODS))
    parser.add_argument("--beta_min", default=0.02, type=float)
    parser.add_argument("--beta_max", default=4, type=float)
    parser.add_argument("--N", default=200, type=int)
    parser.add_argument("--noise_steps", default=3, type=int)
    parser.add_argument("--snr", default=0.0, type=float)
    parser.add_argument("--noise_path", type=str)
    parser.add_argument("--noise_snr", default=0, type=float)
    parser.add_argument("--noise_dbfs", default=-20, type=int)
    parser.add_argument("--noise_gain", default=1, type=float)
    parser.add_argument("--defense_model_path")
    parser.add_argument("--config_path", default=None)
    parser.add_argument("--classifier_path")
    parser.add_argument("--zero_mean", default=False, type=bool)
    parser.add_argument("--n_fft", default=400, type=int)
    parser.add_argument("--hop_size", default=100, type=int)
    parser.add_argument("--win_size", default=400, type=int)
    parser.add_argument("--compress_factor", default=0.3, type=float)

    # DiffWave-VPSDE (AudioPure) arguments
    parser.add_argument("--ddpm_config", type=str, default=config.DEFAULT_DDPM_CONFIG,
                        help="JSON file for the DiffWave configuration")
    parser.add_argument("--ddpm_path", type=str, default=config.DEFAULT_DDPM_PATH,
                        help="DiffWave checkpoint used by the AudioPure defense")
    parser.add_argument("--sample_step", type=int, default=1, help="Total sampling steps")
    parser.add_argument("--t", type=int, default=3,
                        help="Diffusion steps controlling the sampling noise scale")
    parser.add_argument("--t_delta", type=int, default=0,
                        help="Perturbation range of the sampling noise scale (0 disables it)")
    parser.add_argument("--rand_t", action="store_true", default=False,
                        help="Randomise the sampling noise scale")
    parser.add_argument("--diffusion_type", type=str, default="ddpm", help="[ddpm, sde]")
    parser.add_argument("--score_type", type=str, default="guided_diffusion",
                        help="[guided_diffusion, score_sde, ddpm]")
    parser.add_argument("--use_bm", action="store_true", default=False,
                        help="Use brownian motion")

    # generative-defense checkpoints
    parser.add_argument("--defense_gan_path", default=None, type=str)
    parser.add_argument("--consistency_path", default=None, type=str)

    # attack configuration
    parser.add_argument("--steps", default=50, type=int)
    parser.add_argument("--attack_type", default="pgd", choices=list(ATTACK_TYPES))
    parser.add_argument("--epsilon", default=8 / 255, type=float)
    parser.add_argument("--alpha", default=2 / 255, type=float)
    parser.add_argument("--bound_norm", default="linf", choices=["linf", "l2"])
    parser.add_argument("--eot_size", default=1, type=int)
    return parser
