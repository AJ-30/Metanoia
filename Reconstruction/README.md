# Reconstruction: diffusion-based completion of missing floor-plan components

Self-contained package for the project *Diffusion-Based Reconstruction of
Missing Architectural Floor Plan Components*. It does **not** import or modify
the `VAE/` and `Diffusion/` experiments elsewhere in this repository.

Task setting: at test time the model gets only a floor plan with one component
missing (wall, door, window, room or other element) and must generate that
component. Where it is missing and what type it is are unknown to the model.

## Status

| Milestone | State |
|---|---|
| 1. Data: ResPlan + CubiCasa5K loading, rasterization, wall splitting, splits, leakage control | done (`data_prep/`) |
| 2. Missing-component generation, frozen val/test cases, comparison visualizer | done (`data_prep/`) |
| 3. Evaluation: component extraction, matching, metrics, baselines | done (`evaluation/`) |
| 4+. Diffusion model, constraints, learned baselines, ablations | next |

## Layout

```
Reconstruction/
├── README.md            this file
├── requirements.txt     dependencies
├── setup_env.sh         installs them into a conda env
├── common/              TOML config loading, logger
├── data_prep/           milestones 1-2 (see data_prep/README.md)
├── evaluation/          milestone 3 (see evaluation/README.md)
└── tests/               unit tests (synthetic data, no dataset needed)
```

## Quick start (from the repository root)

```bash
bash Reconstruction/setup_env.sh metanoia            # once
conda activate metanoia

# Build the processed datasets (add --mode debug for a 200-item smoke test)
python -m Reconstruction.data_prep.build_dataset --dataset resplan --mode prod
python -m Reconstruction.data_prep.build_dataset --dataset cubicasa --mode prod

# Before/after comparison grids
python -m Reconstruction.data_prep.visualize --dataset resplan --mode prod --split test

# Evaluate a baseline (oracle / empty / rules)
python -m Reconstruction.evaluation.evaluate --dataset resplan --predictor rules --mode prod

# Tests
python -m pytest Reconstruction/tests -q
```

Training code will use `data_prep.dataset.MissingComponentDataset`:

```python
from pathlib import Path
from Reconstruction.data_prep.dataset import MissingComponentDataset

train = MissingComponentDataset(Path("datasets/processed/resplan"), "train", seed=42)
train.set_epoch(epoch)       # new random missing component per plan each epoch
item = train[0]              # incomplete / target / full / mask tensors + metadata
```
