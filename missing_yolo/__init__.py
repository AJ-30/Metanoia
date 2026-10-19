"""missing_yolo: train a detector for the MISSING component of a layout.

This stage trains Ultralytics YOLO26 to localise exactly one furniture
component (the "missing component": ``tv``, ``showcase`` or ``dining_set``)
in images produced by the frozen ``mask_diffusion`` generator. Each run targets
one component, chosen by the ``MISSING_COMPONENT`` hyperparameter (or the
``--component`` CLI override):

    floorplan_segmentation.png ─┐
                                ├─► frozen mask diffusion ─► z ─► frozen decoder
    furniture_segmentation.png ─┘   (component erased)      │        │
    (labels.txt)  ── component boxes ───────────────────────┴─► YOLO26 training target

The image is the floorplan render with the GENERATED arrangement blended over
it, and the detection target is the dataset's own ``labels.txt`` box for that
component - never a box derived from the generated pixels. Runs are fully
self-contained: ``missing_yolo/config.toml`` holds one row per component
(generator checkpoint, class name, ``labels.txt`` names) and the three runs
never share a dataset tree.
"""