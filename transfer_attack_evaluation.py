#!/usr/bin/env python
"""Transfer-attack evaluation.

Adversarial examples are crafted on a *surrogate* classifier and then
transferred to the real victim, measured both with and without the
speech-enhancement defense in front of it.
"""
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from se_eval import (
    build_parser,
    collate,
    get_attack,
    get_classifier,
    get_dataset,
    get_defense_module,
    get_surrogate_classifier,
)
from defense_modules import AcousticSystem


def parse_args():
    parser = build_parser(description=__doc__)
    parser.add_argument(
        "--surrogate_classifier_path",
        default="/data/nas07/PersonalData/apoman123/m18_sc09_zero_mean/m5net_81.pt",
        type=str,
    )
    parser.add_argument("--surrogate_zero_mean", default=False, type=bool)
    return parser.parse_args()


@torch.no_grad()
def standard_predictions(modules, loader, device):
    """Clean-input predictions for each module in ``modules`` (name -> callable)."""
    preds = {name: [] for name in modules}
    labels_all = []
    for batch in tqdm(loader):
        values = batch["input_values"].to(device)
        labels_all += list(batch["labels"])
        for name, module in modules.items():
            preds[name] += torch.argmax(module(values), dim=1).detach().cpu()
    return preds, labels_all


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    dataset = get_dataset(args)
    loader = DataLoader(
        dataset, batch_size=args.batch_size, collate_fn=collate,
        num_workers=args.num_workers, pin_memory=args.pin_memory,
    )

    classifier = get_classifier(args)
    surrogate_classifier = get_surrogate_classifier(args)
    model = AcousticSystem(get_defense_module(args), classifier)
    for module in (classifier, surrogate_classifier, model):
        module.to(device)
        module.eval()

    # attack the surrogate classifier
    atk = get_attack(args, surrogate_classifier)

    # sanity check: clean accuracy of every branch
    modules = {
        "defended": model,
        "vanilla": classifier,
        "surrogate": surrogate_classifier,
    }
    std_preds, std_labels = standard_predictions(modules, loader, device)
    for name, preds in std_preds.items():
        print(f"{name} standard_acc: {accuracy_score(preds, std_labels) * 100}%")

    # transfer the adversarial examples to the victim
    robust_preds = {"defended": [], "vanilla": [], "surrogate": []}
    robust_labels = []
    for batch in tqdm(loader):
        values = batch["input_values"].to(device)
        labels = batch["labels"].to(device)
        adv_values, _ = atk.generate(x=values, y=labels)

        with torch.no_grad():
            robust_preds["defended"] += torch.argmax(model(adv_values), dim=1).detach().cpu()
            robust_preds["vanilla"] += torch.argmax(classifier(adv_values), dim=-1).detach().cpu()
        robust_preds["surrogate"] += torch.argmax(
            surrogate_classifier(adv_values), dim=-1
        ).detach().cpu()
        robust_labels += labels.cpu()

    print(f"vanilla robust_acc: {accuracy_score(robust_preds['vanilla'], robust_labels) * 100}%")
    print(f"defended robust_acc: {accuracy_score(robust_preds['defended'], robust_labels) * 100}%")


if __name__ == "__main__":
    main()
