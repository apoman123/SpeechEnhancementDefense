#!/usr/bin/env bash
#
# Example invocations for the unified evaluation CLI.
#
# These are templates: replace the checkpoint/config paths with your own (see the
# "Model weights" section of the README) and pick the GPU you want to use.
# They are not meant to be run end-to-end; copy the block you need.
set -euo pipefail

export CUDA_VISIBLE_DEVICES=0

# Dataset roots can also be pointed at your own copies via environment variables:
#   export SEAE_SC09_PATH=/path/to/sc09
#   export SEAE_VCTK_PATH=/path/to/vctk_mfcc_with_speaker_labels
#   export SEAE_QKWS_PATH=/path/to/QKWS

# ---------------------------------------------------------------------------
# 1. Single white-box PGD evaluation of the MP-SENet defense on SC09.
# ---------------------------------------------------------------------------
python attack_evaluation.py \
    --dataset sc09 --batch_size 8 \
    --defense_module MPSENet --noise_adding_method GaussianDBFS --noise_dbfs -32 \
    --config_path /path/to/mpsenet/config.json \
    --defense_model_path /path/to/mpsenet/g_00185000 \
    --classifier_path /path/to/m18net_sc09.pt \
    --attack_type pgd --bound_norm linf --steps 70 --epsilon 0.002

# ---------------------------------------------------------------------------
# 2. Sweep the PGD step budget (replaces the old attack_multiple_evaluation.py).
# ---------------------------------------------------------------------------
python attack_evaluation.py \
    --dataset sc09 --batch_size 8 \
    --defense_module AudioPure --classifier_path /path/to/m18net_sc09.pt \
    --attack_type pgd --epsilon 0.002 \
    --sweep steps --sweep_values 10,20,30,50,70,100

# ---------------------------------------------------------------------------
# 3. Sweep EOT size (replaces attack_eot_evaluation.py).
# ---------------------------------------------------------------------------
python attack_evaluation.py \
    --dataset sc09 --batch_size 8 \
    --defense_module MPSENet --noise_adding_method VPSDE \
    --config_path /path/to/mpsenet/config.json \
    --defense_model_path /path/to/mpsenet/g_00185000 \
    --classifier_path /path/to/m18net_sc09.pt \
    --attack_type pgd --epsilon 0.002 --steps 70 \
    --sweep eot --sweep_values 1,5,10,15,20,25

# ---------------------------------------------------------------------------
# 4. Sweep injected-noise loudness in dBFS, writing a CSV
#    (replaces attack_multiple_dbfs.py / find_dbfs_with_attack.py).
# ---------------------------------------------------------------------------
python attack_evaluation.py \
    --dataset qkws --batch_size 2 \
    --defense_module MPSENet --noise_adding_method GaussianDBFS \
    --config_path /path/to/mpsenet/config.json \
    --defense_model_path /path/to/mpsenet_qkws/g_00185000 \
    --classifier_path /path/to/kwsmodel_qkws/model.safetensors \
    --attack_type pgd --steps 50 --epsilon 0.002 \
    --sweep dbfs --sweep_values=-20,-25,-30,-35 --output qkws_dbfs.csv

# ---------------------------------------------------------------------------
# 5. Black-box FakeBob attack on the speaker-ID (VCTK) task.
# ---------------------------------------------------------------------------
python attack_evaluation.py \
    --dataset vctk --batch_size 256 \
    --defense_module Identity --classifier_path /path/to/xvector_vctk/model.safetensors \
    --attack_type fakebob --epsilon 0.002

# ---------------------------------------------------------------------------
# 6. Multi-GPU evaluation (4 GPUs) of the same configuration as example 1.
# ---------------------------------------------------------------------------
torchrun --nproc_per_node=4 attack_evaluation_multiprocess.py \
    --dataset sc09 --batch_size 8 \
    --defense_module MPSENet --noise_adding_method GaussianDBFS --noise_dbfs -32 \
    --config_path /path/to/mpsenet/config.json \
    --defense_model_path /path/to/mpsenet/g_00185000 \
    --classifier_path /path/to/m18net_sc09.pt \
    --attack_type pgd --steps 70 --epsilon 0.002
