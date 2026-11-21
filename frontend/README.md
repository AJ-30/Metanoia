# frontend — upload a floor plan, see the generated component and the detector's boxes

Deliverable 2's interface: the part of the project a person can actually talk to.
Choose the **missing component** (`tv`, `showcase`, `dining_set`), upload the
floor-plan class-id mask of a plan that is missing it, and the page shows

1. the floorplan you uploaded, rendered in the layout's colours;
2. the layout the generator was conditioned on (your furniture, with the missing
   component folded into the background exactly as the model was trained);
3. the ensemble component the diffusion generated, in its own colour, on its own;
4. that component placed into your layout;
5. **what YOLO makes of it** — the same composite with the detector's bounding
   boxes drawn, plus a table of class, confidence and box coordinates.

The primary result is an ensemble: `DEFAULT_DRAWS` independent chains are
generated, their missing-component softmax probabilities are averaged per pixel,
and the average is thresholded at `ENSEMBLE_PROBABILITY_THRESHOLD`. Every chain
is also shown as a per-draw diagnostic section.

If you also upload the layout's ground-truth YOLO `labels.txt`, the page draws
the ground truth in **green** and the predictions in **red**, adds a ground-truth
only overlay, and reports IoU, TP/FP/FN, precision, recall, F1, Jaccard, centre
distance, box error and overlap with the other named furniture for every draw.

The component selector drives **both** models: the dropdown chooses a row of
`components.toml` at the repository root, which pairs the component's
`mask_diffusion` checkpoint with the `missing_yolo` detector trained on that
component. The per-component detectors are single-class, so the component is
class id `0` in the drawn boxes. Paths and the default selection live in
`frontend/config.toml`; every number lives in `frontend/hyperparameters.toml`.

The page also links to a separate **inpainting mode** (`/inpaint`): erase pixels
of the three component classes with a brush, then run all three component models
one at a time and merge their reconstructions. See section 4.

---

## 0. Step 0: complete the plan first (`/complete`)

A plan with a missing wall, door or window can be completed before the furniture step. `frontend/completion.py` runs the wall/door/window
model (`Reconstruction/serve`, a FastAPI app) inside this process, mounts its API under `/api` and serves its page at `/complete`. The page's
**Next: furniture** step (`complete/handoff_step.js`) posts the completed plan to `/handoff`, which converts it with `handoff.py`
(class ids and output size in `handoff.toml`) to `floorplan_segmentation.png` and `furniture_segmentation.png`, stores them under
`uploads/handoff_<id>/` and opens `/?plan=<id>` or `/inpaint?plan=<id>` with both files already chosen. The furniture pages themselves are
the same with or without this step. The plan model's default checkpoint and profiles are configured in `Reconstruction/serve/`.

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

The first upload of a component builds its context — the three frozen mask
VAEs, that component's diffusion denoiser and its detector — which takes a few
seconds. Each component's context is cached separately, so switching back to a
component that was already used is instant. Inpainting loads all three component
contexts because it runs every model; that is the memory cost of the merged
mode. `Ctrl-C` stops it. Nothing is trained, and nothing outside
`frontend/uploads` and `frontend/outputs` is written.

## 2. What to upload

| Field | Required | Content |
|---|---|---|
| **Missing component** | yes | Dropdown of every row in `components.toml`; selects the generator **and** the detector together. |
| **Floor-plan mask** | yes | `floorplan_segmentation.png` — single-channel class-id PNG (mode `L`), floorplan classes 0–17. Any pixel size; it is nearest-neighbour resized to the model's native square size. |
| **Furniture mask** | no | `furniture_segmentation.png` — the layout the component has to fit into. Leave empty to generate into an empty floor plan. Its classes are 0–10; any instance of the selected component already in it is ignored, because that is exactly what the model was trained on. |
| **Ground-truth labels** | no | `labels.txt` — the layout's boxes for the selected component. One `cls cx cy w h` line per box (numeric YOLO class id), one `Name cx cy w h` line (the dataset's own form, e.g. `TV 0.5 0.5 0.2 0.2`), or `cx cy w h` when the class is implicit; all values normalized to 0–1 on the same square render domain. A numeric class id must match the selected detector's component class; named lines for other furniture are not scored as the target but are used by the overlap metric, so the dataset's multi-class `labels.txt` can be uploaded unchanged. Blank lines and an empty file mean “no component here”. |

An upload that is not a class-id mask (a colour render, a JPEG photo, a random
file) is refused on the form with a message about the upload, never a stack trace
and never a server path. The same guard catches a class id above the layout's
range instead of silently clamping it onto a neighbouring class. A malformed
label file gets the same treatment: the line number and the problem are shown on
the form, and the diffusion chain never starts.

**Ensemble members (diffusion draws)**: `4` is the default; up to `MAX_DRAWS`
(4) independent chains are averaged into the primary result. Each chain is a
full 1000-step reverse chain, which is why the count is bounded, and every chain
still gets its own diagnostic panel set below the ensemble.

## 3. Reading the result page

* The page starts with the **ensemble** result, then repeats the same panel
  layout for every draw as a diagnostic. The first four panels are the
  floorplan, the conditioned layout, the generated component and the untouched
  composite.
* With ground-truth labels, panel 5 is the composite with **green** ground-truth
  boxes only, and panel 6 is the same composite with ground truth in green and
  the detector's predictions in **red**. Without labels there is no panel 5 and
  panel 6 shows red predictions only.
* The metrics block is computed per result on the 256×256 composite; the
  ensemble's metrics come first, and each draw's metrics follow. A prediction
  claims a ground-truth box when their IoU is at least `MATCH_IOU_THRESHOLD`
  (`frontend/hyperparameters.toml`, `0.5` by default); predictions are taken in
  confidence order and each box can be claimed once. TP is a match, FP is an
  unmatched prediction, FN is an unmatched ground-truth box.
* Precision = `TP/(TP+FP)`, recall = `TP/(TP+FN)`, F1 = `2PR/(P+R)`, Jaccard =
  `TP/(TP+FP+FN)`. “Mean IoU over union” divides the sum of matched IoUs by the
  union count, so FP and FN boxes contribute zero.
* **Overlap with other GT furniture** asks whether the generated component was
  placed on top of furniture that is already in the layout. It is the fraction
  of the detector's highest-confidence target-component box covered by the
  union of the other named furniture boxes in the uploaded labels. `0` is best
  (no collision), `1` means the predicted box is completely covered. `n/a`
  means no target box was predicted; `0` with a prediction means the upload had
  no other named furniture to collide with. The metric needs the dataset's
  named `Name cx cy w h` labels to identify other furniture; a numeric
  target-only YOLO file has no other boxes, so it reports `0`.
* The localization table reports mean, median, p10, p90, min and max of the
  matched IoUs, centre distances (pixels and normalized by the render side) and
  the L1 error of the four normalized box parameters.
* `n/a` means a rate is undefined (for example recall with no ground-truth
  boxes), which is deliberately different from a real zero. With no labels
  uploaded, nothing is scored and the page says so instead of pretending every
  prediction is a false positive.
* The prediction table's centre and size are normalized to `0..1` in the
  256×256 render domain — the same convention as a YOLO label file.
* “generated pixels” is the size of the generated component mask; a layout where
  the model produced nothing shows `0` and the detector will have nothing to find.

## 4. Inpainting mode

`/inpaint` is a separate route with its own editor; it does not change the
single-component upload flow. Upload the floor-plan mask and the original
furniture mask; the page normalizes the furniture mask to the model's square
class-id domain. The canvas then shows the layout through the furniture palette:

* pick a brush size and drag over **TV**, **showcase** or **dining-set** pixels;
* erased pixels become background class `0` and are shown in translucent red;
* **Undo** restores the previous stroke and **Clear erasures** restores the
  uploaded mask;
* furniture outside the three component classes cannot be erased, and the
  server re-checks that rule before any model runs.

On submit, all three component models run **one at a time**. Each component
uses the same ensemble as the normal page: `DEFAULT_DRAWS` chains, averaged
missing-component softmax probabilities, one threshold at
`ENSEMBLE_PROBABILITY_THRESHOLD`. The model never receives the erase mask; it
only sees the edited furniture mask with erased pixels at background.

The result page shows the input layout, the erase overlay, each component's
generated mask and merged mask, each component's composite and detector boxes,
a small per-draw diagnostic strip for every component, then one merged layout
with all three detectors. The merge rule is

```
final_component = remaining_original OR generated
```

With `INPAINT_PROTECT_ORIGINAL = true` (the default), generated pixels may only
fill background, so a generated TV cannot overwrite a sofa that survived the
erasure. When two generated components claim the same background pixel, the
component with the higher ensemble probability wins; a tie keeps the first
component in `components.toml` order. Set `INPAINT_PROTECT_ORIGINAL = false` to
follow the literal union rule and let generated pixels claim any pixel that is
not that component's remaining original.

Inpainting does not upload or score ground-truth labels; the result page reports
per-component generated/preserved pixel counts and the detectors' boxes on both
the per-component and merged composites.

## 5. Package layout

| File | Responsibility |
|---|---|
| `server.py` | Flask routes: form, upload handler, inpainting editor/run, image servers, run pruning |
| `settings.py` | Per-component context building and caching (one load per component) |
| `pipeline.py` | Upload → validate → encode → sample → average probabilities → threshold → composite → detect → compare; plus the three-component inpainting pipeline |
| `inpainting.py` | Pure inpainting rules: legal-edit validation, palette loading, component merge and overlap resolution |
| `comparison.py` | Parse uploaded YOLO labels, match them to predictions, compute IoU/rates/localization and overlap with other furniture |
| `../components.toml` | Component → diffusion checkpoint + `missing_yolo` detector, shared with the evaluator |
| `templates/index.html` | The single-component upload form |
| `templates/result.html` | The panels, ground-truth comparison, metrics and detection table |
| `templates/inpaint.html` | The canvas editor, brush controls and palette |
| `templates/inpaint_result.html` | Per-component reconstructions and the merged layout |
| `test_frontend.py` | Unit tests of the pure logic (`./venv/bin/python frontend/test_frontend.py`) |

Everything the stage knows about generation and detection is imported, not
reimplemented: `mask_diffusion/inference.py` (encode, sample, decode),
`evaluator/compositor.py` (the image the detector judges), `evaluator/yolo_eval.py`
(the detector call) and `evaluator/visualize.py` (box drawing). The only things
this stage owns are the upload handling and the two pages.

## 6. Configuration

`config.toml` picks the default component, the directories and the device;
`hyperparameters.toml` picks the numbers (draws, ensemble threshold, chunk,
alpha, confidence, size caps, match IoU, label-file guards, cached runs, and the
inpainting canvas/brush/overlay/merge settings). The component -> models pairing
is **not** declared here: `components.toml` at the repository root is the single
source the evaluator and the frontend both read, so the two stages can never
disagree about which generator and detector belong together. To add a component,
add one row there (and its label names to `COMPONENT_LABEL_NAMES`); no Python
edit is needed.
