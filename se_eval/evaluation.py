"""Reusable evaluation loops (clean / defended / robust accuracy)."""
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score
from tqdm import tqdm

from . import config


def _predict(forward, loader, device):
    """Run ``forward`` over ``loader`` and collect (predictions, labels)."""
    preds, targets = [], []
    for batch in tqdm(loader):
        values = batch["input_values"].to(device)
        labels = batch["labels"]
        logits = forward(values)
        preds += torch.argmax(logits, dim=1).detach().cpu()
        targets += labels.cpu()
    return preds, targets


def clean_accuracy(classifier, loader, device):
    """Accuracy of the bare classifier on un-perturbed inputs."""
    classifier.eval()
    with torch.no_grad():
        preds, targets = _predict(
            lambda x: F.softmax(classifier(x), dim=-1), loader, device
        )
    return accuracy_score(preds, targets) * 100


def defended_accuracy(model, loader, device, defense_module=None):
    """Accuracy of the full ``defense -> classifier`` pipeline on clean inputs.

    ``DefenseGAN`` reconstructs its input through an inner optimisation loop that
    needs gradients, so it is the one defense evaluated without ``no_grad``.
    """
    model.eval()
    if defense_module == "DefenseGAN":
        preds, targets = _predict(model, loader, device)
    else:
        with torch.no_grad():
            preds, targets = _predict(model, loader, device)
    return accuracy_score(preds, targets) * 100


def robust_accuracy(model, attack, loader, device, dataset=None):
    """Accuracy of the full pipeline against adversarial inputs from ``attack``."""
    model.eval()
    if dataset in config.TRAIN_MODE_DATASETS:
        model.train()

    preds, targets = [], []
    for batch in tqdm(loader):
        values = batch["input_values"].to(device)
        labels = batch["labels"].to(device)
        adv_values, _ = attack.generate(x=values, y=labels)
        # No ``no_grad`` here: DefenseGAN reconstructs its input through an inner
        # optimisation loop that requires gradients even at inference time.
        logits = model(adv_values)
        preds += torch.argmax(logits, dim=1).detach().cpu()
        targets += labels.cpu()
    return accuracy_score(preds, targets) * 100
