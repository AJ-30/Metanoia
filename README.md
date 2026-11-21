# Metanoia: reconstructing missing components of a floor plan with diffusion models

Given a living-room plan that is incomplete, the system completes it in two stages, each a trained diffusion model:

| | Method | Adds | Dataset | Code |
|---|---|---|---|---|
| 1 | **Immovable components**: a conditional pixel-space diffusion U-Net redraws the one missing wall, door or window (the ResPlan model also rooms and stairs) | wall, door, window (ResPlan: also room, stair, ...) | FurnitureArranger (and ResPlan) | [`Reconstruction/`](Reconstruction/README.md) |
| 2 | **Movable furniture**: mask diffusion in the latent space of frozen mask VAEs, plus one YOLO detector per piece | TV, showcase, dining set | FurnitureArranger | `mask_VAE/`, `mask_diffusion/`, `missing_yolo/` and the packages they build on (map below) |
| 1 + 2 | **Combined demo**: remove or miss a wall/door/window, complete the plan with method 1, hand the result to method 2 and place the missing furniture | all of the above | | `frontend/` |

```
incomplete plan --(method 1: Reconstruction)--> complete shell --(frontend/handoff.py)--> two class-id masks
                                                                                              |
                          TV / showcase / dining set  <--(method 2: mask diffusion + YOLO)-----+
```

## Repository map

| Folder | Role |
|---|---|
| `Reconstruction/` | **Method 1.** `data_prep/` (rasterize plans, missing-component cases), `model/` (U-Net, v-prediction, DDIM), `postprocess/`, `evaluation/`, `furniture_model/` (training for the FurnitureArranger models), `serve/` (FastAPI demo of this method alone), `tests/`. Each folder has a README. |
| `mask_VAE/` | **Method 2.** VAEs over floor-plan and furniture masks (frozen once trained) |
| `mask_diffusion/` | **Method 2.** The denoiser that generates a missing piece in the mask-VAE latent space, and `inference.py` (the sampler the demo uses) |
| `missing_yolo/` | **Method 2.** One YOLO detector per piece, trained on the generated masks |
| `evaluator/` | **Method 2.** Scores generation plus detection on the test split |
| `VAE/`, `Diffusion/`, `conditional_diffusion/`, `YOLO/` | Earlier members of the same family that `mask_diffusion` and the evaluator build on (image VAE, unconditional and floor-plan-conditional diffusion, first detector) |
| `frontend/` | **Combined demo.** Flask app: the furniture pages (`/`, `/inpaint`) and step 0, `/complete` (`completion.py`, `handoff.py`, `handoff.toml`) |
| `components.toml` | Pairs each piece with its mask-diffusion checkpoint and detector |
| `Color-index/`, `Val-Samples/` | Class-id tables of the masks; sample plans with masks, labels and the ground-truth SVG |
| `datasets/` | Parameters, size bins and example plans of the three datasets of method 1 (large files are not versioned) |
| `helpers/`, `TrainerBase.py`, `ProjectTrainer.py`, `main.py`, `*.toml`, `*.sh` | Shared logger, trainer base classes, configs and the train / run scripts of method 2 |
| `setup/`, `run_scripts/` | Environment creation and helper scripts |

## Setup

```bash
setup/create_venv.sh          # conda env ./venv from setup/environment.yml (method 2 and the demo)
bash Reconstruction/setup_env.sh archpainting   # separate env for method 1 alone (Blackwell GPUs: TORCH_INDEX_URL=https://download.pytorch.org/whl/cu130)
```

The combined demo needs one environment with both sets of dependencies; `setup/environment.yml` lists them. It needs a CUDA GPU for step 0
(`frontend/config.toml` sets `DEVICE`; use `mps` or `cpu` for the furniture models on other machines).

## Data and weights

Code, configs, example plans and the two serve-only checkpoints of method 1 are in git. Large files are not; they are expected at these paths:

| Path | Content |
|---|---|
| `Reconstruction/serve/weights/furniture_{shell,walls}_step18000.pt` | Method 1, FurnitureArranger models (in git) |
| `Reconstruction/runs/run2-a40/checkpoints/milestone_step_0150000.pt` | Method 1, ResPlan model (the default model of the demo page) |
| `datasets/processed/resplan/test/{structure,room}.npy`, `datasets/resplan/12/ResPlan.pkl` | ResPlan example plans and raw data (see `Reconstruction/README.md`) |
| `mask_VAE/models/`, `mask_diffusion/models/`, `missing_yolo/models/` | Method 2 checkpoints named in `components.toml` and in the `mask_diffusion` README |

## Run the demo

```bash
./run_frontend.sh prod                    # http://127.0.0.1:8087   (or: ./run_frontend.sh prod 0.0.0.0 9000)
```

* `/complete`: load or draw a plan with a missing wall, door or window (try `Val-Samples/<plan>/ground_truth.svg` with a FurnitureArranger
  model, then *Remove and restore*), run the model, then choose **Next: furniture**.
* `/` and `/inpaint`: the furniture pages, opened with the completed plan already chosen. They can also be used alone by uploading
  `floorplan_segmentation.png`, `furniture_segmentation.png` and optionally `labels.txt` of a `Val-Samples/` plan.
* Method 1 alone: `python -m Reconstruction.serve.app --mode prod` (http://127.0.0.1:8090).

Details: [`frontend/README.md`](frontend/README.md), [`Reconstruction/serve/README.md`](Reconstruction/serve/README.md).

## Method 1 in short

Command list in [`Reconstruction/README.md`](Reconstruction/README.md): build the dataset (`data_prep.build_dataset`), train (`model.train`,
`furniture_model.train` for the FurnitureArranger models), predict, post-process, evaluate.

## Method 2 in short

```bash
./train_mask_VAE.sh prod          # frozen mask VAEs
./train_mask_diffusion.sh prod    # denoiser for the missing piece
./train_missing_yolo.sh prod tv   # detector for one piece (tv, showcase, dining_set)
./run_eval.sh prod                # generation plus detection on the test split
```

The demo uses the trained checkpoints directly (nothing is trained when it runs). Ensemble of four draws per piece, merged with the
furniture already in the plan.

## Tests

```bash
python -m pytest Reconstruction/tests -q                        # method 1 (synthetic data)
python -m pytest frontend evaluator missing_yolo test_index_maker.py -q   # method 2 and the demo
```

## Known gaps

* `Reconstruction/furniture_model/` (training of the FurnitureArranger models) needs `config.toml`, `hyperparameters.toml` and
  `data_prep/params_furniture.toml`, which are not in this repository; `Reconstruction/tests/test_furniture.py` depends on them.
  The hyperparameters of the shipped checkpoints are stored inside the `.pt` files.
* `missing_yolo/test_missing_yolo.py::test_read_sample_labels_malformed_component_line_raises` fails (it did before this reorganisation).

## Utilities

### Mask color index

`index_maker.py` turns the dataset's `floorplan_color_index.json` and
`furniture_color_index.json` into readable horizontal, light-mode legend
images and a merged index:

```bash
python index_maker.py
```

Outputs go to `outputs/color_index/`:

| File | Content |
|---|---|
| `index_floorplan.png` | Floorplan classes (rooms, walls, doors, ...) |
| `index_arrangement.png` | Arrangement/furniture classes (sofa, tv, dining set, ...) |
| `index_combined.png` | Both legends in one image |
| `color_index.json` | Both class tables, machine-readable |

Paths, colors and pixel sizes are configured in `index_maker.toml`.
