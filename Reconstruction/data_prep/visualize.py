"""Before/after comparison grids of missing-component cases.

Each row is one case with four panels:
    Before (complete) | After removal (model input) | Missing component | Erased region

Outputs (PNG, in --out-dir or VISUALIZATION_DIR/<dataset>/<split>/):
    * ``<split>_<type>.png``  one grid per component type, easy and hard rows.
    * ``<split>_margin_sweep.png``  the same component at every margin
      (val/test only; uses the frozen margin-sweep cases).

Usage (from the repository root):
    python -m Reconstruction.data_prep.visualize --dataset resplan --mode prod --split test
    python -m Reconstruction.data_prep.visualize --dataset cubicasa --mode prod --split train --rows 8 --types door window
"""

import argparse
from pathlib import Path
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")  # file output only; no display needed on the cluster
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

from Reconstruction.common.config_loader import load_run_config, resolve_path
from Reconstruction.data_prep.case_source import TRAIN_SPLIT, CaseSource
from Reconstruction.data_prep.missing import EASY, HARD, CaseSpec
from Reconstruction.data_prep.render import PANEL_TITLES, case_panels
from Reconstruction.data_prep.schema import Schema

CONFIG_FILE: Path = Path(__file__).resolve().parent / "config.toml"
# Inches per panel in the saved figures.
PANEL_INCHES: float = 3.2
DPI: int = 110
# How many plans are scanned at most when searching train cases on the fly.
MAX_TRAIN_SCAN: int = 5000


def parse_args() -> argparse.Namespace:
    """Command-line arguments.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", required=True, help="Processed dataset name, e.g. resplan or cubicasa")
    parser.add_argument("--mode", choices=["debug", "prod"], default="debug", help="Config table (selects PROCESSED_ROOT)")
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--types", nargs="*", default=None, help="Component types (default: all)")
    parser.add_argument("--rows", type=int, default=6, help="Cases per grid")
    parser.add_argument("--seed", type=int, default=None, help="Sampling seed (default: config SEED)")
    parser.add_argument("--out-dir", type=Path, default=None, help="Output directory")
    return parser.parse_args()


def _candidate_specs(source: CaseSource, comp_type: str, mode: str, margin: int, rng: np.random.Generator) -> list[CaseSpec]:
    """Specs of a split matching (type, mode, margin), in random order.

    Val/test use the frozen cases. Train samples cases on the fly (epoch 0)
    and keeps those that match.

    Args:
        source (CaseSource): Split source.
        comp_type (str): Component type.
        mode (str): Removal mode.
        margin (int): Margin in px.
        rng (np.random.Generator): Order randomization.

    Returns:
        list[CaseSpec]: Matching specs.
    """
    order: np.ndarray = rng.permutation(len(source))
    if source.split == TRAIN_SPLIT:
        order = order[:MAX_TRAIN_SCAN]
    specs = (source.spec(int(i), epoch=0) for i in order)
    return [s for s in specs if s.comp_type == comp_type and s.mode == mode and s.margin_px == margin]


def _row_label(info: dict) -> str:
    """Two-line description of a case shown left of its row.

    Args:
        info (dict): Case metadata.

    Returns:
        str: Label text.
    """
    return (f"plan {info['plan_id']} | {info['primary_cls']} | {info['mode']} | m={info['margin_px']}\n"
            f"size={info['size_bin']} | {info['location']} | nbrs={info['num_neighbours']} | removed={len(info['removed'])}")


def save_grid(source: CaseSource, specs: list[CaseSpec], title: str, path: Path) -> None:
    """Render cases as a grid (one row per case) and save it.

    Args:
        source (CaseSource): Split source used to build the cases.
        specs (list[CaseSpec]): Cases to show, top to bottom.
        title (str): Figure title.
        path (Path): Output PNG path.
    """
    schema: Schema = source.schema
    fig, axes = plt.subplots(len(specs), len(PANEL_TITLES), squeeze=False,
                             figsize=(PANEL_INCHES * len(PANEL_TITLES), PANEL_INCHES * len(specs) + 1.2))
    for row, spec in enumerate(specs):
        case = source.build(spec)
        for col, panel in enumerate(case_panels(case, schema)):
            ax = axes[row, col]
            ax.imshow(panel, interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])
            if row == 0:
                ax.set_title(PANEL_TITLES[col], fontsize=11)
        axes[row, 0].set_ylabel(_row_label(case.info), fontsize=7)
    handles = [Patch(color=schema.colors[name], label=name) for name in schema.channels]
    handles.append(Patch(color=schema.colors["mask_outline"], alpha=0.45, label="erased"))
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), fontsize=8, frameon=False)
    fig.suptitle(title, fontsize=13)
    fig.tight_layout(rect=(0, 0.03, 1, 0.98))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=DPI)
    plt.close(fig)


def type_grids(source: CaseSource, types: list[str], rows: int, rng: np.random.Generator, out_dir: Path) -> list[Path]:
    """One grid per component type: first half easy, second half hard (margin 0).

    Args:
        source (CaseSource): Split source.
        types (list[str]): Component types to draw.
        rows (int): Rows per grid.
        rng (np.random.Generator): Sampling randomness.
        out_dir (Path): Output directory.

    Returns:
        list[Path]: Written files.
    """
    written: list[Path] = []
    hard_types: list[str] = source.params.missing.HARD_MODE_TYPES
    for comp_type in types:
        n_hard: int = rows // 2 if comp_type in hard_types else 0
        specs = _candidate_specs(source, comp_type, EASY, 0, rng)[:rows - n_hard]
        if n_hard:
            specs += _candidate_specs(source, comp_type, HARD, 0, rng)[:n_hard]
        if not specs:
            print(f"No {comp_type} cases in {source.split}; skipped")
            continue
        path: Path = out_dir / f"{source.split}_{comp_type}.png"
        order: str = "easy rows first, then hard" if n_hard else "easy mode"
        save_grid(source, specs, f"{source.split}: missing {comp_type} ({order})", path)
        written.append(path)
    return written


def margin_sweep_grid(source: CaseSource, types: list[str], rng: np.random.Generator, out_dir: Path) -> Path | None:
    """Same component at every margin, one row per margin, for one random type.

    Args:
        source (CaseSource): Val/test source (uses frozen sweep cases).
        types (list[str]): Candidate component types.
        rng (np.random.Generator): Sampling randomness.
        out_dir (Path): Output directory.

    Returns:
        Path | None: Written file, or None if no sweep cases exist.
    """
    margins: list[int] = source.params.missing.MARGINS_PX
    swept: list[CaseSpec] = [s for t in types for s in _candidate_specs(source, t, EASY, max(margins), rng)]
    if not swept:
        return None
    pick: CaseSpec = swept[0]
    specs: list[CaseSpec] = [CaseSpec(**{**pick.__dict__, "margin_px": m}) for m in margins]
    path: Path = out_dir / f"{source.split}_margin_sweep.png"
    save_grid(source, specs, f"{source.split}: missing-region size sweep ({pick.comp_type})", path)
    return path


def main() -> None:
    """Draw the grids (see module docstring)."""
    args: argparse.Namespace = parse_args()
    config: SimpleNamespace = load_run_config(CONFIG_FILE, args.mode)
    seed: int = config.SEED if args.seed is None else args.seed
    source = CaseSource(resolve_path(config.PROCESSED_ROOT) / args.dataset, args.split, seed)
    types: list[str] = args.types or list(source.params.missing.COMPONENT_TYPES)
    out_dir: Path = args.out_dir or resolve_path(config.VISUALIZATION_DIR) / args.dataset / args.split
    rng: np.random.Generator = np.random.default_rng(seed)
    written: list[Path] = type_grids(source, types, args.rows, rng, out_dir)
    if args.split != TRAIN_SPLIT:
        sweep: Path | None = margin_sweep_grid(source, types, rng, out_dir)
        written += [sweep] if sweep else []
    for path in written:
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
