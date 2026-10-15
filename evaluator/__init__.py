"""Evaluator package: judging YOLO on the missing component mask_diffusion generates.

Rebuilds each test sample as the real layout with the missing component erased
plus the component the diffusion generated, lets the trained detector see that
image and scores only that component: the detector's class-filtered metrics, the
box loss against the dataset's own ground-truth box, and the generated channel's
mask Dice/IoU (see ``evaluator/README.md``).
"""
