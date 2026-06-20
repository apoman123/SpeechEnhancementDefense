#!/usr/bin/env python
"""Certified robustness (randomized smoothing) evaluation.

For every test clip we certify a prediction and its robust radius with
:class:`attacks.certified_robust.RobustCertificate`, then dump the per-example
results to ``--save_path/sigma=<sigma>/sigma=<sigma>_N=<num_sampling>.json``.
"""
import json
import os

import torch
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from attacks.certified_robust import RobustCertificate
from defense_modules import AcousticSystem
from se_eval import build_parser, collate, get_classifier, get_dataset, get_defense_module

NUM_CLASSES = {"sc09": 10, "vctk": 108, "qkws": 4}


def parse_args():
    parser = build_parser(description=__doc__)
    parser.add_argument("--save_path", type=str, default="_Experiments/certified_robustness/records")
    parser.add_argument("--sigma", type=float, help="smoothing noise standard deviation")
    parser.add_argument("--num_sampling", default=1000, type=int, help="samples used to certify")
    parser.add_argument("--N0", type=int, default=100, help="samples used to pick the top class")
    return parser.parse_args()


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    dataset = get_dataset(args)
    loader = DataLoader(
        dataset, batch_size=1, collate_fn=collate,
        num_workers=args.num_workers, pin_memory=args.pin_memory,
    )

    model = AcousticSystem(get_defense_module(args), get_classifier(args))
    model.to(device)
    model.eval()
    certifier = RobustCertificate(classifier=model)

    # sanity check: standard accuracy
    preds, targets = [], []
    with torch.no_grad():
        for batch in tqdm(loader):
            values = batch["input_values"].to(device)
            preds += torch.argmax(model(values), dim=1).detach().cpu()
            targets += batch["labels"].cpu()
    print(f"standard_acc: {accuracy_score(preds, targets) * 100}%")

    save_dir = os.path.join(args.save_path, f"sigma={args.sigma}")
    os.makedirs(save_dir, exist_ok=True)
    out_file = os.path.join(save_dir, f"sigma={args.sigma}_N={args.num_sampling}.json")

    records = []
    total = 0
    for batch in tqdm(loader, total=len(loader)):
        x = batch["input_values"].to(device)
        labels = batch["labels"].to(device)

        y_certified, r_certified = certifier.certify(
            x=x, y=labels, sigma=args.sigma, n_0=args.N0,
            n=args.num_sampling, batch_size=args.batch_size,
        )

        for i in range(x.shape[0]):
            records.append({
                "id": i + total,
                "y_true": labels[i].item(),
                "y_pred": y_certified[i].item(),
                "certified_radius": r_certified[i].item(),
            })
        total += x.shape[0]

        with open(out_file, "w") as f:
            json.dump(records, f, indent=4)


if __name__ == "__main__":
    main()
