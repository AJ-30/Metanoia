"""Export the small, versionable assets the demo needs for a trained dataset profile.

Training leaves two things the demo cannot ship as they are: a 500 MB checkpoint
(model, EMA, optimizer) and a processed test split of tens of MB. This tool writes

* a serve-only checkpoint: ``{"step", "hyperparameters", "ema"}`` with the EMA
  weights stored as float16 (about 67 MB for the 33.5M-parameter models). The demo
  loads it with ``weights_only=True``; loading casts the weights back to float32;
* the dataset parameters the demo needs (``params.toml``, ``size_bins.json``);
* an ``examples/`` plan store: an evenly spaced subset of the test plans that are
  not near-duplicates of a training plan (the "open a test plan" buttons).

Usage (from the repository root of this repository, with the training repository's
files at ``--source``):
    python -m Reconstruction.serve.export_assets \
        --checkpoint /path/to/runs/<run>/checkpoints/step_0018000.pt \
        --processed /path/to/datasets/processed/<dataset> \
        --dataset-name furniture_shell --weights-name furniture_shell_step18000.pt --examples 40
"""

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import torch

SERVE_DIR: Path = Path(__file__).resolve().parent
ROOT_DIR: Path = SERVE_DIR.parent.parent  # repository root


def export_checkpoint(source: Path, target: Path) -> None:
    """Write the serve-only checkpoint.

    Args:
        source (Path): Training checkpoint (needs ``ema``, ``hyperparameters``, ``step``).
        target (Path): Output file.
    """
    state = torch.load(source, map_location="cpu", weights_only=False)
    slim = {"step": int(state["step"]), "hyperparameters": state["hyperparameters"],
            "ema": {k: (v.half() if v.is_floating_point() else v) for k, v in state["ema"].items()}}
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.save(slim, target)
    print(f"checkpoint step {slim['step']} -> {target} ({target.stat().st_size / 1e6:.1f} MB)")


def export_examples(processed: Path, target: Path, count: int) -> None:
    """Write an ``examples/`` store from evenly spaced non-duplicate test plans.

    Args:
        processed (Path): Processed dataset directory (with ``test/`` and ``near_duplicates.json``).
        target (Path): Output ``examples`` directory.
        count (int): Number of plans.
    """
    duplicates: set[int] = set()
    dup_file = processed / "near_duplicates.json"
    if dup_file.is_file():
        duplicates = {int(d["eval_index"]) for d in json.loads(dup_file.read_text()).get("test", [])}
    structure = np.load(processed / "test" / "structure.npy", mmap_mode="r")
    room = np.load(processed / "test" / "room.npy", mmap_mode="r")
    records = [json.loads(line) for line in (processed / "test" / "plans.jsonl").read_text().splitlines()]
    keep = [i for i in range(len(records)) if i not in duplicates]
    picks = [keep[int(j)] for j in np.linspace(0, len(keep) - 1, min(count, len(keep))).round()]
    picks = sorted(set(picks))
    target.mkdir(parents=True, exist_ok=True)
    np.save(target / "structure.npy", np.ascontiguousarray(structure[picks]))
    np.save(target / "room.npy", np.ascontiguousarray(room[picks]))
    (target / "plans.jsonl").write_text("".join(json.dumps(records[i]) + "\n" for i in picks))
    print(f"{len(picks)} example plans (of {len(keep)} non-duplicate test plans) -> {target}")


def main() -> None:
    """Entry point (see module docstring)."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--processed", type=Path, required=True)
    parser.add_argument("--dataset-name", required=True, help="Folder name under datasets/processed")
    parser.add_argument("--weights-name", required=True, help="File name under Reconstruction/serve/weights")
    parser.add_argument("--examples", type=int, default=40)
    args = parser.parse_args()
    export_checkpoint(args.checkpoint, SERVE_DIR / "weights" / args.weights_name)
    out: Path = ROOT_DIR / "datasets" / "processed" / args.dataset_name
    out.mkdir(parents=True, exist_ok=True)
    for name in ("params.toml", "size_bins.json"):
        shutil.copyfile(args.processed / name, out / name)
    export_examples(args.processed, out / "examples", args.examples)


if __name__ == "__main__":
    main()
