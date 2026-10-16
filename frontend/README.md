# frontend — upload a floor plan, see the generated component and the detector's boxes

Deliverable 2's interface: the part of the project a person can actually talk to.
Upload the floor-plan class-id mask of a plan that is missing its **TV**, and the
page shows

1. the floorplan you uploaded, rendered in the layout's colours;
2. the layout the generator was conditioned on (your furniture, with the missing
   component folded into the background exactly as the model was trained);
3. the component the diffusion generated, in its own colour, on its own;
4. that component placed into your layout;
5. **what YOLO makes of it** — the same composite with the detector's bounding
   boxes drawn, plus a table of class, confidence and box coordinates.

The detector is the one trained in the `YOLO` stage; the generator is the
`mask_diffusion` run the config points at. Both are configured in
`frontend/config.toml`, every number in `frontend/hyperparameters.toml`.

---

## 1. Running it

```bash
./run_frontend.sh                    # debug mode, http://127.0.0.1:8087
./run_frontend.sh prod               # production logging level
./run_frontend.sh debug 0.0.0.0 9000 # reachable from outside the machine
```

or directly:

```bash
python frontend/server.py --mode debug --host 127.0.0.1 --port 8087
```

The first upload builds the context — the three frozen mask VAEs, the diffusion
denoiser and the detector — which takes a few seconds; later uploads reuse it.
`Ctrl-C` stops it. Nothing is trained, and nothing outside `frontend/uploads` and
`frontend/outputs` is written.

## 2. What to upload

| Field | Required | Content |
|---|---|---|
| **Floor-plan mask** | yes | `floorplan_segmentation.png` — single-channel class-id PNG (mode `L`), floorplan classes 0–17. Any pixel size; it is nearest-neighbour resized to 256. |
| **Furniture mask** | no | `furniture_segmentation.png` — the layout the component has to fit into. Leave empty to generate into an empty floor plan. Its classes are 0–10; any TV already in it is ignored, because that is exactly what the model was trained on. |

An upload that is not a class-id mask (a colour render, a JPEG photo, a random
file) is refused on the form with a message about the upload, never a stack trace
and never a server path. The same guard catches a class id above the layout's
range instead of silently clamping it onto a neighbouring class.

**Diffusion draws**: `1` is the default; up to `MAX_DRAWS` (4) renders one panel
set per draw, so you can see the variation the model has learned. Each draw is a
full 1000-step reverse chain, which is why the count is bounded.

## 3. Reading the result page

* Red boxes are this run's predictions (label + confidence); green boxes would be
  ground truth, and an uploaded floor plan carries no dataset labels, so there are
  none here. For the scored version of this comparison, see `evaluator/README.md`.
* The table's centre and size are normalized to `0..1` in the 256x256 render
  domain — the same convention as a YOLO label file.
* "generated pixels" is the size of the generated component mask; a layout where
  the model produced nothing shows `0` and the detector will have nothing to find.

## 4. Package layout

| File | Responsibility |
|---|---|
| `server.py` | Flask routes: form, upload handler, image servers, run pruning |
| `settings.py` | Context building and caching (one process, one load of the VAE stack) |
| `pipeline.py` | Upload → validate → encode → sample → decode → composite → detect |
| `templates/index.html` | The upload form |
| `templates/result.html` | The five panels and the detection table |
| `test_frontend.py` | Unit tests of the pure logic (`./venv/bin/python frontend/test_frontend.py`) |

Everything the stage knows about generation and detection is imported, not
reimplemented: `mask_diffusion/inference.py` (encode, sample, decode),
`evaluator/compositor.py` (the image the detector judges), `evaluator/yolo_eval.py`
(the detector call) and `evaluator/visualize.py` (box drawing). The only things
this stage owns are the upload handling and the two pages.

## 5. Configuration

`config.toml` picks the diffusion checkpoint, the component, the detector and the
directories; `hyperparameters.toml` picks the numbers (draws, chunk, alpha,
confidence, size caps, cached runs). To judge a different component or run, change
`MISSING_COMPONENT_CLASS_NAME`, `MASK_DIFFUSION_CHECKPOINT_PATH` and
`YOLO_WEIGHTS_PATH` — the label-name table in the hyperparameters is only needed
by the scored evaluator, so no other edit is required here.
