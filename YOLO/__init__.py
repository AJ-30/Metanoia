"""YOLO package: YOLO26 finetuning on live diffusion-drawn floor plans.

Trains a furniture detector on overlays redrawn every epoch from the frozen
floorplan->arrangement diffusion model, with boxes derived from the drawn
arrangement and per-epoch GT-vs-prediction overlays (see ``YOLO/README.md``).
"""
