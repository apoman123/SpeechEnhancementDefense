#!/usr/bin/env python
"""Multi-GPU (DistributedDataParallel) version of :mod:`attack_evaluation`.

Launch with ``torchrun``::

    torchrun --nproc_per_node=4 attack_evaluation_multiprocess.py \\
        --dataset sc09 --defense_module MPSENet ... --attack_type pgd

Predictions are gathered onto rank 0, which reports clean / standard / robust
accuracy.  The model/dataset/attack are built with the same shared factories as
the single-GPU path, so behaviour stays in sync.
"""
import torch
import torch.distributed as dist
import torch.nn.functional as F
from sklearn.metrics import accuracy_score
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from defense_modules import AcousticSystem
from se_eval import (
    build_parser,
    collate,
    get_attack,
    get_classifier,
    get_dataset,
    get_defense_module,
)


def gather_to_rank0(tensor, rank, world_size):
    """Gather equally-sized ``tensor`` from every rank onto rank 0."""
    if rank == 0:
        buffer = [torch.empty_like(tensor) for _ in range(world_size)]
        dist.gather(tensor, gather_list=buffer)
        return torch.cat(buffer, dim=0).cpu()
    dist.gather(tensor)
    return None


def collect(forward, loader, device, rank, world_size, grad=False):
    """Run ``forward`` over the local shard and gather predictions on rank 0."""
    preds, labels = [], []
    context = torch.enable_grad() if grad else torch.no_grad()
    with context:
        for batch in tqdm(loader, disable=rank != 0):
            values = batch["input_values"].to(device)
            target = batch["labels"].to(device)
            preds.append(torch.argmax(forward(values, target), dim=1))
            labels.append(target)
    preds = gather_to_rank0(torch.cat(preds), rank, world_size)
    labels = gather_to_rank0(torch.cat(labels), rank, world_size)
    return preds, labels


def main():
    args = build_parser(description=__doc__).parse_args()
    dist.init_process_group(backend=args.backend)
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = f"cuda:{rank}"
    torch.cuda.set_device(device)

    dataset = get_dataset(args)
    sampler = DistributedSampler(dataset, shuffle=False)
    loader = DataLoader(
        dataset, batch_size=args.batch_size, collate_fn=collate,
        num_workers=args.num_workers, sampler=sampler, pin_memory=args.pin_memory,
    )

    classifier = get_classifier(args).to(device)
    model = AcousticSystem(get_defense_module(args), classifier).to(device)
    model = DDP(model)
    classifier = DDP(classifier)
    attack = get_attack(args, model)

    model.eval()
    classifier.eval()

    clean_preds, clean_labels = collect(
        lambda x, y: F.softmax(classifier(x), dim=-1), loader, device, rank, world_size
    )
    standard_preds, standard_labels = collect(
        lambda x, y: model(x), loader, device, rank, world_size
    )
    robust_preds, robust_labels = collect(
        lambda x, y: model(attack.generate(x=x, y=y)[0]),
        loader, device, rank, world_size, grad=True,
    )

    if rank == 0:
        print(f"original_acc: {accuracy_score(clean_preds, clean_labels) * 100}%")
        print(f"standard_acc: {accuracy_score(standard_preds, standard_labels) * 100}%")
        print(f"robust_acc: {accuracy_score(robust_preds, robust_labels) * 100}%")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
