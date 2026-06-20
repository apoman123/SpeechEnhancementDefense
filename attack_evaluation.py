#!/usr/bin/env python
"""Adversarial robustness evaluation for speech-enhancement defenses.

This single entry point replaces the former family of near-identical scripts
(``attack_multiple_evaluation``, ``attack_eot_evaluation``,
``attack_multiple_dbfs``, ``attack_multiple_gain_evaluation``,
``find_dbfs_with_attack``, ``find_gain_with_attack`` ...).  Their only real
difference was *which axis they swept*, which is now the ``--sweep`` option.

Examples
--------
Single configuration::

    python attack_evaluation.py --dataset sc09 --defense_module MPSENet \\
        --noise_adding_method GaussianDBFS --noise_dbfs -32 \\
        --config_path config.json --defense_model_path g_00185000 \\
        --classifier_path m18net.pt --attack_type pgd --steps 70 --epsilon 0.002

Sweep PGD steps::

    python attack_evaluation.py ... --sweep steps

Sweep the injected-noise loudness (requires ``--noise_adding_method GaussianDBFS``)::

    python attack_evaluation.py ... --sweep dbfs --output sc09_dbfs.csv
"""
import csv

import torch
from torch.utils.data import DataLoader

from se_eval import (
    build_acoustic_system,
    build_parser,
    clean_accuracy,
    collate,
    defended_accuracy,
    get_attack,
    get_dataset,
    robust_accuracy,
)

SWEEP_DEFAULTS = {
    "steps": [10, 20, 30, 50, 70, 100],
    "eot": [1, 5, 10, 15, 20, 25],
    "dbfs": list(range(-20, -40, -1)),
    "gain": [i / 20 for i in range(1, 21)],
}


def parse_args():
    parser = build_parser(description=__doc__)
    parser.add_argument(
        "--sweep", default="none", choices=["none", "steps", "eot", "dbfs", "gain"],
        help="Axis to sweep while attacking (default: a single configuration).",
    )
    parser.add_argument(
        "--sweep_values", default=None, type=str,
        help="Comma-separated overrides for the swept axis (default depends on --sweep).",
    )
    parser.add_argument(
        "--output", default=None, type=str,
        help="Optional CSV file to write the (sweep value, accuracies) table to.",
    )
    return parser.parse_args()


def sweep_values(args):
    values = SWEEP_DEFAULTS[args.sweep]
    if args.sweep_values:
        cast = float if args.sweep == "gain" else int
        values = [cast(v) for v in args.sweep_values.split(",")]
    return values


def set_noise_level(model, axis, value):
    """Mutate the in-place noise injector for a ``dbfs`` / ``gain`` sweep."""
    noise_adding = getattr(model.defense_module, "noise_adding", None)
    if noise_adding is None:
        raise ValueError(
            f"--sweep {axis} requires a defense with an injected-noise stage "
            "(e.g. MPSENet with --noise_adding_method GaussianDBFS/GaussianZeroMean)."
        )
    setattr(noise_adding, "target_dbfs" if axis == "dbfs" else "gain", value)


def attack_for(args, model, value):
    """Build the attack for one point of a ``steps`` / ``eot`` sweep."""
    if args.sweep == "steps":
        return get_attack(args, model, steps=value)
    if args.sweep == "eot":
        return get_attack(args, model, eot_size=value)
    return get_attack(args, model)


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    dataset = get_dataset(args)
    loader = DataLoader(
        dataset, batch_size=args.batch_size, collate_fn=collate,
        num_workers=args.num_workers, pin_memory=args.pin_memory,
    )

    model = build_acoustic_system(args)
    model.to(device)
    classifier = model.classifier

    print(f"original_acc: {clean_accuracy(classifier, loader, device)}%")

    rows = []
    if args.sweep in ("none", "steps", "eot"):
        std = defended_accuracy(model, loader, device, args.defense_module)
        print(f"standard_acc: {std}%")

        if args.sweep == "none":
            atk = get_attack(args, model)
            robust = robust_accuracy(model, atk, loader, device, args.dataset)
            print(f"robust_acc: {robust}%")
            rows.append({"sweep_value": "", "standard_acc": std, "robust_acc": robust})
        else:
            robust_accs = []
            for value in sweep_values(args):
                atk = attack_for(args, model, value)
                acc = robust_accuracy(model, atk, loader, device, args.dataset)
                robust_accs.append(acc)
                rows.append({"sweep_value": value, "standard_acc": std, "robust_acc": acc})
            print(f"robust_accs ({args.sweep}={sweep_values(args)}): {robust_accs} in %")
    else:  # dbfs / gain: the injected noise changes, so re-measure standard too
        for value in sweep_values(args):
            set_noise_level(model, args.sweep, value)
            std = defended_accuracy(model, loader, device, args.defense_module)
            atk = get_attack(args, model)
            robust = robust_accuracy(model, atk, loader, device, args.dataset)
            print(f"{args.sweep}={value}: standard_acc={std}% robust_acc={robust}%")
            rows.append({"sweep_value": value, "standard_acc": std, "robust_acc": robust})

    if args.output:
        with open(args.output, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["sweep_value", "standard_acc", "robust_acc"])
            writer.writeheader()
            writer.writerows(rows)
        print(f"wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
