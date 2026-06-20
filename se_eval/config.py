"""Central configuration for the evaluation pipeline.

All machine-specific paths used to be hard-coded and copy-pasted across a dozen
scripts.  They are gathered here so the project can be moved between machines by
setting a handful of environment variables instead of editing source files.

Override any default by exporting the matching environment variable, e.g.::

    export SEAE_VCTK_PATH=/mnt/data/vctk_mfcc_with_speaker_labels
"""
import os

# ---------------------------------------------------------------------------
# Datasets (Hugging Face ``datasets`` saved with ``save_to_disk``)
# ---------------------------------------------------------------------------
DATASET_PATHS = {
    "sc09": os.environ.get(
        "SEAE_SC09_PATH", "/data/nas07/SharedBySMB/apoman123/sc09"
    ),
    "vctk": os.environ.get(
        "SEAE_VCTK_PATH",
        "/data/nas07/PersonalData/apoman123/vctk_mfcc_with_speaker_labels",
    ),
    "qkws": os.environ.get(
        "SEAE_QKWS_PATH", "/data/nas07/PersonalData/apoman123/QKWS_10DBFS_LOUDER"
    ),
}

# ---------------------------------------------------------------------------
# Default checkpoint locations (only used as argparse defaults; pass the
# corresponding CLI flag to point at your own copy).
# ---------------------------------------------------------------------------
DEFAULT_DDPM_CONFIG = os.environ.get(
    "SEAE_DDPM_CONFIG", "./diffusion_models/DiffWave_Unconditional/config.json"
)
DEFAULT_DDPM_PATH = os.environ.get(
    "SEAE_DDPM_PATH", "./diffusion_models/unconditional_diffwave.pkl"
)

# Target sampling rate for every audio dataset.
SAMPLE_RATE = 16000

# Datasets whose victim classifier must run in ``train`` mode while attacking
# (their RNN layers disable cuDNN double-backward in eval mode).
TRAIN_MODE_DATASETS = {"qkws"}
