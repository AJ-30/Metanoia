"""Unit tests for the mask color index maker.

The tests use tiny temporary index files rather than the real dataset, so they
run on any machine and cover the logic that can silently mislabel a legend:

1. class ids must stay contiguous from 0 (a gap would shift every color);
2. RGB triples must serialize as lowercase ``#rrggbb``;
3. the combined image must stack both legends without rescaling;
4. the merged JSON must keep both groups separate.

Run with: ``./venv/bin/python test_index_maker.py``
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

ROOT_DIR: Path = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from PIL import Image

from index_maker import (
    IndexEntry,
    IndexStyle,
    build_json_payload,
    combine_legends,
    draw_legend,
    load_entries,
    rgb_to_hex,
)


def make_style() -> IndexStyle:
    """Build a small deterministic style for drawing tests.

    Returns:
        IndexStyle: A compact style with fixed colors.
    """
    return IndexStyle(
        swatch_size=20,
        card_gap=6,
        label_gap=3,
        font_size=8,
        title_font_size=10,
        padding=6,
        section_gap=10,
        min_width=120,
        background=(255, 255, 255),
        text=(17, 17, 17),
        muted=(75, 75, 75),
        border=(160, 160, 160),
    )


def write_index(path: Path, rows: dict[str, tuple[str, str]]) -> None:
    """Write a minimal ``*_color_index.json`` file.

    Args:
        path (Path): Destination JSON path.
        rows (dict[str, tuple[str, str]]): Class id -> ``(name, hex color)``.
    """
    payload = {
        class_id: {"name": name, "color": color}
        for class_id, (name, color) in rows.items()
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_rgb_to_hex_formats_lowercase_six_digits() -> None:
    """A color triple becomes the same ``#rrggbb`` string the dataset uses."""
    assert rgb_to_hex((244, 114, 182)) == "#f472b6"
    assert rgb_to_hex((0, 0, 0)) == "#000000"


def test_load_entries_reads_contiguous_index() -> None:
    """Names, colors and ids are copied in class-id order."""
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "furniture_color_index.json"
        write_index(
            path,
            {
                "0": ("background", "#000000"),
                "1": ("sofa", "#f472b6"),
                "2": ("tv", "#a78bfa"),
            },
        )
        entries = load_entries(path)
        assert [entry.class_id for entry in entries] == [0, 1, 2]
        assert [entry.name for entry in entries] == ["background", "sofa", "tv"]
        assert entries[1].hex_color == "#f472b6"


def test_load_entries_rejects_non_contiguous_ids() -> None:
    """A missing id would shift every later color, so it must fail."""
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "bad_color_index.json"
        write_index(path, {"1": ("sofa", "#f472b6")})
        try:
            load_entries(path)
        except KeyError as exc:
            assert "no gaps" in str(exc)
        else:
            raise AssertionError("A non-contiguous color index must be rejected.")


def test_draw_legend_and_combine_keep_both_sections() -> None:
    """The combined image is the two legends plus the configured gap."""
    style = make_style()
    floorplan = [
        IndexEntry(0, "background", (0, 0, 0), "#000000"),
        IndexEntry(1, "living", (217, 217, 217), "#d9d9d9"),
    ]
    arrangement = [
        IndexEntry(0, "background", (0, 0, 0), "#000000"),
        IndexEntry(1, "sofa", (244, 114, 182), "#f472b6"),
    ]
    top = draw_legend(floorplan, "Floorplan", "floorplan.json", style)
    bottom = draw_legend(arrangement, "Arrangement", "furniture.json", style)
    combined = combine_legends(top, bottom, style)
    assert isinstance(combined, Image.Image)
    assert top.width >= style.min_width and bottom.width >= style.min_width
    # The request is a horizontal legend, so both sections must be wider than
    # they are tall even with this deliberately small two-entry fixture.
    assert top.width > top.height and bottom.width > bottom.height
    expected = (
        max(top.width, bottom.width),
        top.height + style.section_gap + bottom.height,
    )
    assert combined.size == expected


def test_build_json_payload_keeps_groups_and_sources() -> None:
    """The merged JSON carries both tables and their source paths."""
    floorplan = [IndexEntry(0, "background", (0, 0, 0), "#000000")]
    arrangement = [IndexEntry(0, "background", (0, 0, 0), "#000000")]
    payload = build_json_payload(
        Path("/data/floorplan_color_index.json"),
        Path("/data/furniture_color_index.json"),
        floorplan,
        arrangement,
    )
    assert payload["floorplan"]["source"].endswith("floorplan_color_index.json")
    assert payload["arrangement"]["source"].endswith("furniture_color_index.json")
    assert payload["floorplan"]["classes"]["0"]["color"] == "#000000"


def _run_all_tests() -> int:
    """Run every ``test_*`` function in this module.

    Returns:
        int: Number of failures, ``0`` when all tests pass.
    """
    tests = [
        (name, function)
        for name, function in sorted(globals().items())
        if name.startswith("test_") and callable(function)
    ]
    failures: list[str] = []
    for name, function in tests:
        try:
            function()
        except Exception as exc:  # noqa: BLE001 - report and continue
            failures.append(f"{name}: {exc}")
            print(f"FAIL {name}: {exc}")
        else:
            print(f"pass {name}")
    print(f"\n{len(tests) - len(failures)}/{len(tests)} tests passed.")
    return len(failures)


if __name__ == "__main__":
    raise SystemExit(1 if _run_all_tests() else 0)
