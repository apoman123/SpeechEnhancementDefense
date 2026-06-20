# Speech-Enhancement Adversarial Evaluation
Accepted by ICASSP 2026: "Adversarial Defense via Generative Speech Enhancement Module"
Benchmarks for studying whether **speech-enhancement / generative-purification
front-ends** make audio classifiers more robust to adversarial examples.

The pipeline puts a defense module in front of a victim classifier:

```
waveform ──▶ [ noise injection ] ──▶ [ speech-enhancement / purification ] ──▶ [ classifier ] ──▶ label
            └──────────────── defense module ─────────────────┘
```

and reports three numbers:

| Metric          | Meaning                                                             |
| --------------- | ------------------------------------------------------------------- |
| `original_acc`  | bare classifier on clean audio (no defense)                         |
| `standard_acc`  | full *defense → classifier* pipeline on clean audio                 |
| `robust_acc`    | full pipeline on adversarial audio crafted against it               |

## Tasks, datasets and victim classifiers

| Task                          | Dataset | Victim model                     |
| ----------------------------- | ------- | -------------------------------- |
| Speech-command classification | `sc09`  | `M18` / `M5` (`victim_models/`)  |
| Speaker identification        | `vctk`  | x-vector (`victim_models/xvector.py`) |
| Keyword spotting              | `qkws`  | RCNN `KWSModel` (`victim_models/RCNN.py`) |

## Defenses

Wired into the CLI (`--defense_module`):

- `MPSENet` – MP-SENet magnitude/phase speech enhancement
- `AudioPure` – DiffWave diffusion purification (VP-SDE reverse process)
- `Consistency` – consistency-model purification on a log-mel spectrogram
- `DefenseGAN` – WaveGAN-based reconstruction defense
- `AS`, `MS`, `AT` – time-domain simple defenses (`simple_defenses/`)
- `DS`, `LPF`, `BPF` – frequency-domain simple defenses
- `Identity` – no defense (baseline)

Randomized noise injected before enhancement (`--noise_adding_method`):
`VPSDE`, `Gaussian`, `GaussianDBFS`, `GaussianZeroMean`, `Background`,
`VPSDEBackground`, `Identity`.

## Attacks

- `pgd` – white-box PGD (`attacks/pgd.py`, supports EOT via `--eot_size`)
- `fakebob` – black-box FakeBob (`attacks/fakebob.py`)
- `torchattacks_pgd` – PGD from the `torchattacks` library
- Certified robustness via randomized smoothing (`attacks/certified_robust.py`)

## Repository layout

```
attack_evaluation.py               # main CLI (single run + parameter sweeps)
attack_evaluation_multiprocess.py  # multi-GPU (torchrun) version
transfer_attack_evaluation.py      # surrogate → victim transfer attacks
certified_robustness_evaluation.py # randomized-smoothing certification
calc_inference_time.py             # latency benchmark for each defense
se_eval/                           # shared library used by every entry point
  ├── config.py        # dataset / checkpoint paths (env-var overridable)
  ├── builders.py      # dataset, classifier, defense, attack factories
  ├── evaluation.py    # clean / defended / robust accuracy loops
  └── cli.py           # shared argument parser
defense_modules.py                 # defense wrappers + feature extractors
attacks/                           # attack implementations
victim_models/                     # victim classifiers + their training scripts
*_components/                      # vendored model architectures (CMGAN, MP-SENet,
                                   #   SEMamba, Mamba-SEUNet, DefenseGAN, consistency, ...)
diffusion_models/                  # DiffWave / improved-diffusion for AudioPure
scripts/run_examples.sh            # copy-paste example invocations
```

## Installation

```bash
# 1. Install PyTorch for your CUDA version (see https://pytorch.org).
# 2. Install the rest:
pip install -r requirements.txt
```

Some defenses pull extra dependencies (DeepFilterNet, Mamba, etc.); see the
commented "optional" section in `requirements.txt`.

## Datasets

Datasets are loaded with Hugging Face `datasets.load_from_disk`. Point the
pipeline at your local copies with environment variables (defaults live in
`se_eval/config.py`):

```bash
export SEAE_SC09_PATH=/path/to/sc09
export SEAE_VCTK_PATH=/path/to/vctk_mfcc_with_speaker_labels
export SEAE_QKWS_PATH=/path/to/QKWS
```

## Model weights

**No checkpoints are committed to this repository** (they are large binaries and
are excluded by `.gitignore`). Provide your own and pass them via the CLI flags
below.

| Component                | CLI flag                 | How to obtain it                                                                 |
| ------------------------ | ------------------------ | -------------------------------------------------------------------------------- |
| Victim: SC09 (M18 / M5)  | `--classifier_path`      | Train with `victim_models/train_m18.sh` / `train_m5.sh`                          |
| Victim: VCTK (x-vector)  | `--classifier_path`      | Train an x-vector speaker classifier on VCTK (`victim_models/xvector.py`)        |
| Victim: QKWS (RCNN)      | `--classifier_path`      | Train with `victim_models/train_RCNN.sh`                                          |
| MP-SENet enhancer        | `--defense_model_path` + `--config_path` | Train / download from the [MP-SENet](https://github.com/yxlu-0102/MP-SENet) project |
| AudioPure DiffWave       | `--ddpm_path` + `--ddpm_config` | Train with `diffusion_models/DiffWave_Unconditional/`, or use a [DiffWave](https://github.com/philsyn/DiffWave-unconditional) checkpoint |
| Consistency model        | `--consistency_path`     | Train with `consistency_components/consistency_models/` (after fetching the [upstream repo](https://github.com/cloneofsimo/consistency_models)) |
| DefenseGAN / WaveGAN     | `--defense_gan_path`     | Train a [WaveGAN](https://github.com/mostafaelaraby/wavegan-pytorch) generator   |

The AudioPure (`Consistency`) defense also downloads a HiFi-GAN vocoder
automatically via `torch.hub` (`bshall/hifigan`).

> **Maintainer note:** if you publish trained checkpoints, attach them to a
> GitHub Release or a Hugging Face model repo and replace the "How to obtain it"
> column above with direct download links.

## Usage

Single white-box PGD evaluation:

```bash
python attack_evaluation.py \
    --dataset sc09 --batch_size 8 \
    --defense_module MPSENet --noise_adding_method GaussianDBFS --noise_dbfs -32 \
    --config_path /path/to/mpsenet/config.json \
    --defense_model_path /path/to/mpsenet/g_00185000 \
    --classifier_path /path/to/m18net_sc09.pt \
    --attack_type pgd --bound_norm linf --steps 70 --epsilon 0.002
```

### Parameter sweeps

A single `--sweep` flag replaces the family of near-identical scripts that used
to exist (`attack_multiple_evaluation.py`, `attack_eot_evaluation.py`,
`attack_multiple_dbfs.py`, `find_*_with_attack.py`, ...):

| `--sweep` | Sweeps over           | Default grid (override with `--sweep_values`) |
| --------- | --------------------- | --------------------------------------------- |
| `none`    | nothing (single run)  | —                                             |
| `steps`   | PGD iterations        | `10,20,30,50,70,100`                          |
| `eot`     | EOT ensemble size     | `1,5,10,15,20,25`                             |
| `dbfs`    | injected-noise dBFS¹  | `-20 … -39`                                   |
| `gain`    | zero-mean noise gain¹ | `0.05 … 1.0`                                  |

¹ `dbfs` requires `--noise_adding_method GaussianDBFS`; `gain` requires
`GaussianZeroMean`.

```bash
# sweep PGD steps
python attack_evaluation.py ... --sweep steps

# sweep injected-noise loudness and save a CSV
python attack_evaluation.py ... --noise_adding_method GaussianDBFS \
    --sweep dbfs --sweep_values=-20,-25,-30,-35 --output results.csv
```

### Multi-GPU

```bash
torchrun --nproc_per_node=4 attack_evaluation_multiprocess.py --dataset sc09 ...
```

See `scripts/run_examples.sh` for more ready-to-edit examples.
