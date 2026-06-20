"""Shared building blocks for the speech-enhancement adversarial evaluation.

The public helpers below are re-exported so entry-point scripts can simply do::

    from se_eval import build_parser, get_dataset, collate, build_acoustic_system
"""
from .builders import (
    AttrDict,
    build_acoustic_system,
    build_classifier,
    collate,
    dict2namespace,
    get_attack,
    get_classifier,
    get_dataset,
    get_defense_module,
    get_noise_module,
    get_surrogate_classifier,
    load_config,
    short_audio,
)
from .cli import build_parser
from .evaluation import clean_accuracy, defended_accuracy, robust_accuracy

__all__ = [
    "AttrDict",
    "build_acoustic_system",
    "build_classifier",
    "build_parser",
    "clean_accuracy",
    "collate",
    "defended_accuracy",
    "dict2namespace",
    "get_attack",
    "get_classifier",
    "get_dataset",
    "get_defense_module",
    "get_noise_module",
    "get_surrogate_classifier",
    "load_config",
    "robust_accuracy",
    "short_audio",
]
