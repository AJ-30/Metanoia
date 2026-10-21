"""Evaluator package: judging a detector on the missing component diffusion generates.

The evaluated detector is selected together with the generator: ``--component``
reads ``components.toml`` and pairs the component's mask_diffusion checkpoint with
the ``missing_yolo`` detector trained for that component.

Rebuilds each test sample as the real layout with the missing component erased
plus the component the diffusion generated, lets the trained detector see that
image and scores only that component: the detector's class-filtered metrics, the
box loss against the dataset's own ground-truth box, and the generated channel's
mask Dice/IoU (see ``evaluator/README.md``).
"""
