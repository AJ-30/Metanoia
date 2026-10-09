# mask_diffusion — generating a missing furniture component (the TV)

This package learns to generate the **missing channel of a missing furniture
component**: given a floorplan and a layout whose TV was never drawn, it produces
a full arrangement **including** the TV, in a frozen mask-VAE latent space.

It is the third member of the project's latent-space family:

| Package | Task | Conditions | Target space |
|---|---|---|---|
| `Diffusion` | unconditional arrangement | none | image-VAE latent |
| `conditional_diffusion` | floorplan → arrangement | 1 latent | mask-VAE latent |
| **`mask_diffusion`** | **missing component (TV) generation** | **2 latents** | **mask-VAE latent** |

---

## 1. The idea

Three FROZEN `mask_VAE` checkpoints define the pipeline. None of them is trained
here; each is rebuilt from the hyperparameters embedded in its own checkpoint, so
these paths stay loadable after the `mask_VAE` config moves on.

| Role | Run id | File | What it does |
|---|---|---|---|
| spatial condition | `mint-akita` | `mask_vae_mint-akita_floorplan_epoch_200.pt` | encodes `floorplan_segmentation.png` (18 classes) |
| TV-free condition | `fair-burro` | `mask_vae_fair-burro_no_tv__epoch_310.pt` | encodes the furniture mask **with the TV erased** (11 channels, `furniture/tv` excluded) |
| diffusion space + decoder | `chief-yeti` | `mask_vae_chief-yeti_epoch_arrangement_555.pt` | encodes the furniture mask **keeping the TV**, and decodes a latent back into 11 mask channels |

All three share `LATENT_DIM = 128` and `IMAGE_SIZE = 256`.

### How the TV is removed

The trick is that **the partial and the target encoder see the identical
class-id PNG** — only the *layout* differs:

```
furniture_segmentation.png (class ids 0..10)
        │
        ├──► partial layout  (fair-burro)  tv excluded ──► remap_ids_to_background ──► z_partial
        └──► target  layout  (chief-yeti)  tv kept                                  ──► z_target
```

`build_frozen_mask_vae` reads the exclusion straight out of the fair-burro
checkpoint (`mask_excluded_classes = {'furniture': ['furniture/tv']}`) and turns it
into `excluded_local_ids`, so `encode_masks_to_single_channel` → `remap_ids_to_background`
does the erasure. **There is no pixel-editing code in this package.** The only
information separating condition from target is the missing component, which is
exactly the task.

Measured, the two latent spaces are genuinely distinct: `mean |z_target − z_partial| ≈ 2.09`
against `mean |z| ≈ 2.1`, i.e. the partial latent really does not already contain
the TV. There is no leakage through the condition.

---

## 2. Objective

Two objectives, run **in sequence and never summed within an epoch**
(`LOSS_SCHEDULE`).

### `mse` — latent MSE (most epochs)

```
t ~ U[0, NUM_TIMESTEPS);   x_t = q_sample(z_target, t, ε)
loss = MSE( denoiser(x_t, t, z_floorplan, z_partial), ε )
```

Standard DDPM noise prediction (Ho et al. 2020). Two properties make it the
right default here:

- **The frozen decoder is not in the graph at all.** Memory is dominated by the
  128-d latents, so `MSE_BATCH_SIZE` is 16× larger than the pixel epoch's.
- **Timesteps span the full schedule.** Truncating was only ever necessary
  because a *decoded* BCE is meaningless at high noise; a latent objective is
  well defined at every noise level, so the "trains `t<300` but sampling starts
  at 1000" mismatch **disappears in this phase**.

### `bce` — pixel BCE (one epoch per cycle)

```
x0_hat = denormalize( (x_t − sqrt(1−ᾱ_t)·ε̂) / sqrt(ᾱ_t) )
logits = chief_yeti.decoder(x0_hat)                # frozen, 11×256×256
loss   = TV_LOSS_WEIGHT · BCE(logits[:, tv], gt[:, tv])
```

**Only the missing component's channel is scored.** The other ten furniture
classes get no pixel-space term at all, for two reasons:

- the latent MSE has no notion of *which* class matters — it treats the whole
  128-d latent equally;
- a whole-mask BCE is dominated by the ~97% of pixels that are background, which
  are trivially predicted. Measured on a real epoch the two terms were nearly the
  same magnitude (`tv_bce 0.0084`, the whole-mask term 0.0091 — both since
  removed from the hyperparameters), so a 0.1 weight had been needed just to stop
  the easy background half from taking half the gradient. Dropping the term
  entirely is cleaner than down-weighting it.

Since the deliverable is the missing component, the pixel objective supervises
exactly that channel.

The decoder is frozen but **not** under `no_grad`, so the pixel-space gradient
flows back through it into the latent and into the denoiser. That is why this
epoch runs at a small batch size, and why its timesteps stay capped at
`BCE_TRAIN_T_MAX` — a decoded BCE is only informative while `x0_hat` is still
near the latent manifold (`ᾱ=0.396` at `t=300`).

`gt` is the one-vs-rest expansion of the true class-id mask. The channel index is
**resolved by name** (`MISSING_COMPONENT_CLASS_NAME = "tv"` → `furniture/tv` →
channel 2), never hardcoded.

> **Consequence worth knowing:** on `bce` epochs the other ten classes receive no
> gradient at all — only the TV channel's logits are in the loss. They are not
> degraded (nothing touches them) but they are not corrected either; the latent
> MSE is their only supervision.

### The default schedule: `mse_bce_cycle`

`MSE_EPOCHS_PER_CYCLE` latent epochs, then **one** pixel epoch, repeating for the
whole run:

```
MSE  MSE  MSE  MSE  MSE  BCE │ MSE  MSE  MSE  MSE  MSE  BCE │ …
 1    2    3    4    5    6 │ 7    8    9   10   11   12 │ …
```

With the defaults (5, 1200 epochs) that is 200 pixel epochs spread evenly across
the run, so the component keeps being supervised directly instead of being
aligned once at the end and then slowly forgotten by 1000 latent epochs.

Alternatives: `mse_then_bce` (one switch, no cycling), `mse_only` (plain latent
DDPM), `bce_only` (the original objective everywhere).

### Why this order

A pure decoded BCE has no ε-MSE anchor, so it can push `x0_hat` off-manifold,
and it cannot train the high-noise half of the schedule at all. Leading with the
latent MSE gives a well-behaved, fast, full-schedule model; the pixel epoch then
spends one expensive epoch re-aligning the output with what the frozen decoder
actually rewards — the TV.

Verified: latent MSE falls **0.968 → 0.223** (77%) over 30 epochs on 4k samples,
with test MSE 0.192.

> If the component placement is still weak, lower `MSE_EPOCHS_PER_CYCLE` — every
> pixel epoch is expensive (~5.6 min vs ~0.16 min), so this is the direct dial
> for how much pixel-space attention the TV gets. `LOSS_SCHEDULE = "mse_only"`
> isolates whether a problem is in the pixel phase or the model itself.

---

## 3. Files

```
mask_diffusion/
├── __init__.py
├── run.py                    entry point: python mask_diffusion/run.py --mode debug|prod
├── model.py                  MaskComponentDiffusion (denoiser + DDPM schedule)
├── runner.py                 MaskDiffusionRunner (data, loss, eval, grid, loop)
├── config.toml               paths, frozen checkpoints, missing-component name
├── hyperparameters.toml      architecture, schedule, loss weights, intervals
└── README.md
```

```
train_mask_diffusion.sh                     launch training (archives old artifacts)
run_scripts/start_mask_diffusion_server.sh  dashboard on port 8083
```

### Outputs

| Artifact | Pattern |
|---|---|
| checkpoint | `mask_diffusion/models/mask_diffusion_<run>_epoch_<N>.pt` |
| telemetry | `mask_diffusion/telemetry_logs/<run>.csv` |
| per-class telemetry | `mask_diffusion/telemetry_logs/<run>_per_class.csv` |
| visualization grid | `mask_diffusion/outputs/reconstruction_epoch_<N>_<run>.png` |

The grid name deliberately starts with `reconstruction_epoch_` so the shared
dashboard in `helpers/server.py` picks it up with **no server change**.

---

## 4. The visualization grid

`NUM_VIZ_SAMPLES` columns, `3 + NUM_GEN_DRAWS` rows:

```
row 1              ground truth — the full arrangement, colour-coded
row 2              floorplan + the GROUND-TRUTH arrangement (the complete answer)
row 3              floorplan + the INPUT layout — every furniture channel the
                   model is given, i.e. all classes EXCEPT the missing component
rows 4 … 3+D       floorplan + ONLY the model's predicted component, one row per
                   fresh diffusion draw
```

Rows 3 and 4+ read together as **"given" + "generated" = the answer**, so a column
can be checked against its row 2 in one glance.

Row 3 is the conditioning itself, not a restatement of row 2: it is rendered with
the same `remap_ids_to_background` rule the partial (fair-burro) encoder applies,
so the class the model has to invent is *visibly* absent while the other nine
furniture classes are all present. Without this row the grid never showed what
the model was actually given, since row 2 still contains the TV it never sees.

Two details make the prediction block readable:

- **The base is the bare floorplan, which contains no furniture at all.** Nothing
  ground-truth can be confused with the model's output.
- **Only the TV channel is drawn.** The model's other furniture is not shown, so a
  misplaced TV is judged against the real room layout rather than against the
  model's own (possibly wrong) sofa.

An earlier version painted the prediction over a ground-truth layout *with the TV
kept in place*. That was unreadable: the model paints the TV in the same colour
the ground truth already occupies there, so a **perfect prediction and a total
miss were pixel-identical** (measured: 0 differing pixels) and the block could
only ever reveal false positives.

A grid is saved at **epoch 0, before any training**, so a wiring bug shows up as
obviously wrong rows rather than as a slow-learning curve.

---

## 5. Telemetry

`telemetry_logs/<run>.csv`

| column | meaning |
|---|---|
| `train_loss` | the objective actually active this epoch |
| `train_mse` | latent MSE — filled on every `mse` epoch |
| `train_tv_bce` | the TV-channel BCE — filled on every `bce` epoch |
| `test_mse` | held-out latent MSE, evaluated **every epoch** |
| `test_tv_bce` | pixel terms, only on epochs a pixel eval ran |
| `test_tv_dice` | TV Dice on the truncated path |
| `sample_tv_dice`, `sample_tv_iou` | **full 1000-step reverse chain** |

A metric that was not measured is written as an **empty cell**, never `0.0` —
the established pattern in this repo. With the cycling schedule this makes the
rhythm directly readable: `train_mse` filled, `train_tv_bce` empty, five times,
then the reverse once, repeating. The log line also marks every alternation with
`Epoch 6: objective 'bce' (batch 16)`. `test_mse` is measured every epoch (no
decoder needed), so the latent curve is unbroken across the whole run.

> **`test_tv_dice` is not a generation metric.** It is measured on the truncated
> training path, where `x0_hat` is a light perturbation of the true latent, so it
> starts near 0.97 before training. `sample_tv_dice` — the full reverse chain from
> pure noise — is the number that reflects whether generation actually works.

`telemetry_logs/<run>_per_class.csv` carries IoU and Dice for **every** channel,
because an averaged loss hides which class is holding the model back.

---

## 6. Caching

The three convolutional encoders run exactly once per split. Each cached sample is
the triple of standardized latents (3 KB) plus the resized uint8 class-id mask
(64 KB). The masks are stored as a **memory-mapped `.npy`** so the training loop
never decodes a PNG and never needs the array resident.

- Three files per split: the latent `.pt`, the furniture class ids and the
  floorplan class ids (both uint8 `(N, 256, 256)`).
- Full-scale train cache ≈ **8.4 GB**, test ≈ 1.2 GB.
- The cache also stores each kept sample's source folder, because the TV filter
  and `MAX_SAMPLES` mean the cached rows are **not** a prefix of the split listing.
- First run encodes at ≈ 200 samples/s on MPS: ~6 min train, ~50 s test. One time.
- A fingerprint over all three checkpoint paths/mtimes/sizes, both split dirs,
  `IMAGE_SIZE`, `USE_MU_ONLY`, `MAX_SAMPLES`, the missing-component filter and
  the resulting counts invalidates the cache automatically. Delete
  `mask_diffusion/latent_cache/` to force a re-encode.
- The cache is **git-ignored** (`.gitignore`: `*.npy`, `latent_cache/`). It is
  regenerable, so committing several GB of it would be pointless.

---

## 7. Data facts worth knowing

Measured over 400 train samples at `IMAGE_SIZE = 256`:

- **89% of samples contain a TV** (356/400). The other 11% would contribute a
  pure-negative TV BCE term, so they are dropped by
  `REQUIRE_MISSING_COMPONENT_SAMPLES = true`. Set it to `false` to keep them.
- **TV is ~3.0% of pixels** (median 3.1%, min 1.9%, max 4.7%) — not razor-sparse.
  That is why `TV_BCE_POS_WEIGHT` defaults to a mild 2.0 rather than 5.0.
- Sample discovery filters `is_dir()`, which drops the stray `.DS_Store` present
  in the split folders.

---

## 8. Reproducibility

`SEED` seeds `random` and `torch`. `USE_MU_ONLY = true` takes the encoder's
posterior mean `mu` (deterministic) rather than a reparameterized sample, so a
rerun with the same seed reproduces the same latents. Checkpoints store the
hyperparameters, the normalization statistics, the resolved missing channel and
all three frozen checkpoint paths, so a checkpoint is self-describing.

---

## 9. Running it

```bash
./train_mask_diffusion.sh          # debug mode
./train_mask_diffusion.sh prod     # production mode

# dashboard (separate port so it can run alongside the others)
./run_scripts/start_mask_diffusion_server.sh      # http://127.0.0.1:8083
```

First launch spends a few minutes building the latent cache; later launches reuse
it.

### Cost, measured on MPS over the full 64k train split

| objective | batch | s/step | epoch |
|---|---|---|---|
| **mse (default)** | **256** | **0.039** | **0.16 min** |
| mse | 512 | 0.045 | 0.09 min |
| mse | 1024 | 0.052 | 0.06 min |
| bce | 16 | 0.084 | 5.6 min |
| bce | 32 | 0.131 | 4.4 min |

The latent epoch is ~35× cheaper, and a 5+1 cycle therefore costs ~6.4 min
dominated by its single pixel epoch. With the defaults (1200 epochs, cycle 5) a
full run is roughly **200 × 6.4 min ≈ 21 h**.

Pixel evaluation also decodes, so on latent epochs it is throttled to every
`PIXEL_EVAL_INTERVAL` epochs; a pixel epoch always evaluates it because that IS
the objective. The latent MSE is scored every epoch since it needs no decoder.
Full reverse-chain sampling (`SAMPLE_INTERVAL`, grid saving) is `NUM_TIMESTEPS`
sequential passes and is dominated by MPS per-op launch overhead rather than math.

While iterating, lower `MAX_SAMPLES` in `hyperparameters.toml` (e.g. `8000`) and
raise it for the final run.