# LatentRxnFlow

Official repository for **Driving Reaction Trajectories via Latent Flow
Matching**.

This repository contains the LatentRxnFlow training and evaluation code derived
from the NERF reaction-prediction codebase, plus the data loader and config
files needed to run it.

## Setup

The reproduction stack is Python 3.7. Create it with conda:

```bash
conda env create -f environment.yml
conda activate latentrxnflow-py37
```

`requirements.txt` is provided for pip-based environments, but conda is
recommended because RDKit is more reliable there:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Data

Large data and checkpoint files are not committed. Prepare NERF-format
preprocessed pickle files and point the config to them:

```yaml
data:
  train_pickle_path: /path/to/train_reactions.pkl
  eval_pickle_path: /path/to/eval_reactions.pkl
```

Alternatively, set `data.pickle_path` and let the training script split one
pickle by `train_ratio` and `eval_ratio`. See `data/README.md` for the expected
fields.

## Training

Single GPU:

```bash
python train_latentrxnflow.py --config configs/base.yaml
```

Multi GPU:

```bash
torchrun --nproc_per_node=4 train_latentrxnflow.py --config configs/base.yaml --ddp
```

Resume training:

```bash
python train_latentrxnflow.py --config configs/base.yaml --resume checkpoints/flow_nerf_baseline_last.pt
```

## Evaluation

Set `eval.checkpoint_path` and `eval.pickle_path` in `configs/eval.yaml`, then
run:

```bash
python eval_multigpu.py --config configs/eval.yaml
```

or with DDP:

```bash
torchrun --nproc_per_node=4 eval_multigpu.py --config configs/eval.yaml --ddp
```

### Legacy epoch 490 reproduction

The NERF epoch 490 checkpoint uses the legacy conditional flow head and
fingerprint condition path. To reproduce that checkpoint, use the provided
configs after placing the checkpoint and pickles locally.

Expected local inputs:

- `checkpoints/flow_nerf_baseline_epoch_490.pt`
- `data/valid_data_rr_only_condition.pickle`
- `data/test_data_rr_only_condition.pickle`

Run the full test evaluation:

```bash
WANDB_MODE=disabled python eval_multigpu.py --config configs/reproduce_epoch490_test_9017.yaml
```

This reproduces the legacy NERF result on
`test_data_rr_only_condition.pickle`:

```text
ode_delta acc = 0.9016955086525957
```

Run validation with:

```bash
WANDB_MODE=disabled python eval_multigpu.py --config configs/reproduce_epoch490_valid.yaml
```

The important compatibility settings are:

- `flow_cond_head: film_residual_add`
- `condition_source: fp`
- `cond_pool: None`
- `eval.nfe: 20`
- `eval.ode_method: heun`

The newer bounded `film_residual` head and condition-embedding path do not
match this checkpoint.

## What Is Not Included

The repository intentionally excludes local artifacts:

- preprocessed reaction pickles
- trained checkpoints
- W&B runs
- experiment output directories

Place those files locally under ignored paths such as `data/`, `checkpoints/`,
and `experiments/`.
