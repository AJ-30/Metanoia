"""Build a visual color index for the dataset's floorplan and arrangement masks.

The mask PNGs store integer class ids, not colors. Colors are defined by two
dataset files:

* ``floorplan_color_index.json`` - rooms, walls, doors, windows, ...
* ``furniture_color_index.json`` - the arrangement's sofa, tv, dining set, ...

Those JSON files are authoritative, but reading hex codes in a text editor is a
poor way to understand a rendered report. This script turns them into:

* one horizontal, light-mode legend PNG per mask family, and one combined
  legend;
* a merged ``color_index.json`` for scripts that want both tables at once;
* a short README pointing at the files.

It reads only the dataset's index files and writes only under
``OUTPUT_DIRECTORY``, so it can be rerun at any time without touching data.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Sequence

# This script lives at the repository root, so its own folder is the root used
# to anchor relative output paths and project imports.
ROOT_DIR: Path = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from PIL import Image, ImageDraw, ImageFont  # noqa: E402

from helpers.config import load_config  # noqa: E402
from helpers.logger import setup_logger  # noqa: E402
from mask_VAE.masking import load_color_index  # noqa: E402

SCRIPT_TOML: str = "index_maker.toml"


@dataclass(frozen=True)
class IndexEntry:
    """One class row of a mask color index.

    Attributes:
        class_id (int): Integer stored in the mask PNG for this class.
        name (str): Human-readable class name from the dataset index.
        color (tuple[int, int, int]): ``(R, G, B)`` display color.
        hex_color (str): The same color as ``#rrggbb``, for labels and JSON.
    """

    class_id: int
    name: str
    color: tuple[int, int, int]
    hex_color: str


@dataclass(frozen=True)
class IndexStyle:
    """Pixel layout and colors of the horizontal legend images.

    Every class is one card in a single left-to-right row: colored swatch on
    top, class id/name underneath, hex value under that.

    Attributes:
        swatch_size (int): Side length of one color square.
        card_gap (int): Horizontal space between neighbouring class cards.
        label_gap (int): Space between a swatch and its first text line.
        font_size (int): Point size of class labels and hex values.
        title_font_size (int): Point size of the section title.
        padding (int): Outer margin around the legend content.
        section_gap (int): Gap between the two bands of the combined image.
        min_width (int): Minimum canvas width before text is measured.
        background (tuple[int, int, int]): Canvas color.
        text (tuple[int, int, int]): Main text color.
        muted (tuple[int, int, int]): Secondary text color.
        border (tuple[int, int, int]): Swatch border color.
    """

    swatch_size: int
    card_gap: int
    label_gap: int
    font_size: int
    title_font_size: int
    padding: int
    section_gap: int
    min_width: int
    background: tuple[int, int, int]
    text: tuple[int, int, int]
    muted: tuple[int, int, int]
    border: tuple[int, int, int]

    @classmethod
    def from_config(cls, config: SimpleNamespace) -> "IndexStyle":
        """Build the style from the script's TOML keys.

        Args:
            config (SimpleNamespace): Parsed ``index_maker.toml``.

        Returns:
            IndexStyle: The configured style.

        Raises:
            ValueError: If a size is not positive, which would make a legend
                impossible to draw.
        """
        style: IndexStyle = cls(
            swatch_size=int(config.SWATCH_SIZE),
            card_gap=int(config.CARD_GAP),
            label_gap=int(config.LABEL_GAP),
            font_size=int(config.FONT_SIZE),
            title_font_size=int(config.TITLE_FONT_SIZE),
            padding=int(config.PADDING),
            section_gap=int(config.SECTION_GAP),
            min_width=int(config.MIN_WIDTH),
            background=(int(config.BACKGROUND_GRAY),) * 3,
            text=(int(config.TEXT_GRAY),) * 3,
            muted=(int(config.MUTED_TEXT_GRAY),) * 3,
            border=(int(config.BORDER_GRAY),) * 3,
        )
        if min(
            style.swatch_size,
            style.card_gap,
            style.label_gap,
            style.font_size,
            style.title_font_size,
            style.padding,
            style.section_gap,
            style.min_width,
        ) <= 0:
            raise ValueError(f"Every index_maker size must be positive, got {style}.")
        return style


@dataclass(frozen=True)
class LegendGeometry:
    """Measured pixel geometry of one horizontal legend section.

    Attributes:
        width (int): Canvas width.
        height (int): Canvas height.
        card_width (int): Width shared by every class card.
        line_height (int): Height of one class-label line.
        title_block (int): Vertical space above the card row.
        card_height (int): Height of the card row itself.
    """

    width: int
    height: int
    card_width: int
    line_height: int
    title_block: int
    card_height: int


def load_entries(color_index_path: Path) -> list[IndexEntry]:
    """Read one dataset color index into drawable entries.

    Args:
        color_index_path (Path): Path of ``*_color_index.json``.

    Returns:
        list[IndexEntry]: One entry per class id, in ascending id order.

    Raises:
        FileNotFoundError: If the index file is absent.
        KeyError: If the index is not contiguous from class id 0.
    """
    names: list[str]
    colors: list[tuple[int, int, int]]
    names, colors = load_color_index(color_index_path)
    return [
        IndexEntry(
            class_id=class_id,
            name=name,
            color=color,
            hex_color=rgb_to_hex(color),
        )
        for class_id, (name, color) in enumerate(zip(names, colors, strict=True))
    ]


def rgb_to_hex(color: tuple[int, int, int]) -> str:
    """Format an RGB triple as a lowercase ``#rrggbb`` string.

    Args:
        color (tuple[int, int, int]): ``(R, G, B)`` values in ``0..255``.

    Returns:
        str: Six-digit hex color, e.g. ``#fb923c``.
    """
    red, green, blue = color
    return f"#{red:02x}{green:02x}{blue:02x}"


def load_font(size: int) -> ImageFont.ImageFont | ImageFont.FreeTypeFont:
    """Load PIL's bundled font at ``size``, with a version-safe fallback.

    Args:
        size (int): Requested point size.

    Returns:
        ImageFont.ImageFont | ImageFont.FreeTypeFont: A usable font object.
    """
    try:
        return ImageFont.load_default(size=int(size))
    except TypeError:
        # Pillow before 10.1 has no sized default font.
        return ImageFont.load_default()


def label_lines(entry: IndexEntry) -> tuple[str, str]:
    """Return the two centered text lines drawn under one class swatch.

    Args:
        entry (IndexEntry): Class card to label.

    Returns:
        tuple[str, str]: ``(class id + name, hex color)``.
    """
    return f"{entry.class_id:>2}  {entry.name}", entry.hex_color


def line_height(font: ImageFont.ImageFont | ImageFont.FreeTypeFont) -> int:
    """Measure a representative text line height for vertical spacing.

    Args:
        font (ImageFont.ImageFont | ImageFont.FreeTypeFont): Font to measure.

    Returns:
        int: Height of ``"Ag"`` in pixels, never below one.
    """
    probe: Image.Image = Image.new("RGB", (1, 1), (0, 0, 0))
    draw: ImageDraw.ImageDraw = ImageDraw.Draw(probe)
    _left, top, _right, bottom = draw.textbbox((0, 0), "Ag", font=font)
    return max(1, int(bottom - top))


def measure_geometry(
    entries: Sequence[IndexEntry],
    title: str,
    subtitle: str,
    style: IndexStyle,
    font: ImageFont.ImageFont | ImageFont.FreeTypeFont,
    title_font: ImageFont.ImageFont | ImageFont.FreeTypeFont,
) -> LegendGeometry:
    """Measure the single-row horizontal legend before any pixels are drawn.

    Every card gets the same width, set by the widest class label, so the row
    stays aligned and long names cannot overlap their neighbours.

    Args:
        entries (Sequence[IndexEntry]): Classes in the row.
        title (str): Section title.
        subtitle (str): Source-file line under the title.
        style (IndexStyle): Paddings, gaps and swatch size.
        font (ImageFont.ImageFont | ImageFont.FreeTypeFont): Class-label font.
        title_font (ImageFont.ImageFont | ImageFont.FreeTypeFont): Title font.

    Returns:
        LegendGeometry: Canvas and card dimensions in pixels.
    """
    probe: Image.Image = Image.new("RGB", (1, 1), style.background)
    draw: ImageDraw.ImageDraw = ImageDraw.Draw(probe)
    heading_width: int = max(
        int(draw.textlength(title, font=title_font)),
        int(draw.textlength(subtitle, font=font)),
    )
    widest_text: int = max(
        (
            int(draw.textlength(text, font=font))
            for entry in entries
            for text in label_lines(entry)
        ),
        default=0,
    )
    card_width: int = max(style.swatch_size, widest_text)
    text_height: int = line_height(font)
    title_block: int = (
        style.title_font_size + style.font_size + 2 * style.padding
    )
    card_height: int = (
        style.swatch_size + style.label_gap + 2 * text_height
    )
    cards_width: int = len(entries) * card_width + max(
        0, len(entries) - 1
    ) * style.card_gap
    width: int = max(
        style.min_width,
        2 * style.padding + cards_width,
        heading_width + 2 * style.padding,
    )
    height: int = title_block + card_height + style.padding
    return LegendGeometry(
        width=width,
        height=height,
        card_width=card_width,
        line_height=text_height,
        title_block=title_block,
        card_height=card_height,
    )


def draw_legend(
    entries: Sequence[IndexEntry],
    title: str,
    subtitle: str,
    style: IndexStyle,
) -> Image.Image:
    """Draw one horizontal, single-row color-index legend.

    Args:
        entries (Sequence[IndexEntry]): Classes to draw, left to right in id
            order.
        title (str): Section title, e.g. ``Floorplan mask colors``.
        subtitle (str): Small line under the title naming the source file.
        style (IndexStyle): Pixel layout and colors.

    Returns:
        Image.Image: RGB legend image.

    Raises:
        ValueError: If ``entries`` is empty.
    """
    if not entries:
        raise ValueError(f"Legend '{title}' has no entries to draw.")
    font = load_font(style.font_size)
    title_font = load_font(style.title_font_size)
    geometry: LegendGeometry = measure_geometry(
        entries, title, subtitle, style, font, title_font
    )
    image: Image.Image = Image.new(
        "RGB", (geometry.width, geometry.height), style.background
    )
    draw: ImageDraw.ImageDraw = ImageDraw.Draw(image)
    draw.text(
        (style.padding, style.padding),
        title,
        fill=style.text,
        font=title_font,
    )
    draw.text(
        (style.padding, style.padding + style.title_font_size + style.label_gap),
        subtitle,
        fill=style.muted,
        font=font,
    )

    card_top: int = geometry.title_block
    for index, entry in enumerate(entries):
        card_left: int = style.padding + index * (
            geometry.card_width + style.card_gap
        )
        swatch_left: int = card_left + (
            geometry.card_width - style.swatch_size
        ) // 2
        draw.rectangle(
            [
                swatch_left,
                card_top,
                swatch_left + style.swatch_size - 1,
                card_top + style.swatch_size - 1,
            ],
            fill=entry.color,
            outline=style.border,
            width=1,
        )
        center_x: float = card_left + geometry.card_width / 2
        name_y: float = (
            card_top
            + style.swatch_size
            + style.label_gap
            + geometry.line_height / 2
        )
        name, hex_color = label_lines(entry)
        draw.text(
            (center_x, name_y),
            name,
            fill=style.text,
            font=font,
            anchor="mm",
        )
        draw.text(
            (center_x, name_y + geometry.line_height),
            hex_color,
            fill=style.muted,
            font=font,
            anchor="mm",
        )
    return image


def combine_legends(
    floorplan: Image.Image,
    arrangement: Image.Image,
    style: IndexStyle,
) -> Image.Image:
    """Stack the two horizontal legend bands without rescaling either.

    Args:
        floorplan (Image.Image): Floorplan legend.
        arrangement (Image.Image): Arrangement legend.
        style (IndexStyle): Provides the gap and background color.

    Returns:
        Image.Image: Combined RGB image, left-aligned.
    """
    width: int = max(floorplan.width, arrangement.width)
    height: int = floorplan.height + style.section_gap + arrangement.height
    combined: Image.Image = Image.new("RGB", (width, height), style.background)
    combined.paste(floorplan, (0, 0))
    combined.paste(arrangement, (0, floorplan.height + style.section_gap))
    return combined


def build_json_payload(
    floorplan_path: Path,
    arrangement_path: Path,
    floorplan: Sequence[IndexEntry],
    arrangement: Sequence[IndexEntry],
) -> dict[str, object]:
    """Build the merged machine-readable index.

    Args:
        floorplan_path (Path): Source floorplan color index.
        arrangement_path (Path): Source arrangement color index.
        floorplan (Sequence[IndexEntry]): Parsed floorplan entries.
        arrangement (Sequence[IndexEntry]): Parsed arrangement entries.

    Returns:
        dict[str, object]: Both index tables plus their source paths.
    """
    def entries_to_table(entries: Sequence[IndexEntry]) -> dict[str, dict[str, str]]:
        """Convert entries to the dataset's id -> {name, color} shape.

        Args:
            entries (Sequence[IndexEntry]): Entries to serialize.

        Returns:
            dict[str, dict[str, str]]: JSON-ready mapping.
        """
        return {
            str(entry.class_id): {"name": entry.name, "color": entry.hex_color}
            for entry in entries
        }

    return {
        "floorplan": {
            "source": str(floorplan_path),
            "classes": entries_to_table(floorplan),
        },
        "arrangement": {
            "source": str(arrangement_path),
            "classes": entries_to_table(arrangement),
        },
    }


def write_readme(
    output_root: Path,
    config: SimpleNamespace,
    floorplan: Sequence[IndexEntry],
    arrangement: Sequence[IndexEntry],
) -> Path:
    """Write a short explanation of the generated index files.

    Args:
        output_root (Path): Output directory.
        config (SimpleNamespace): Parsed ``index_maker.toml``.
        floorplan (Sequence[IndexEntry]): Floorplan entries that were drawn.
        arrangement (Sequence[IndexEntry]): Arrangement entries that were drawn.

    Returns:
        Path: Path of the written README.
    """
    lines: list[str] = [
        "# Mask color index",
        "",
        "Generated by `index_maker.py` from the dataset's own color index files.",
        "",
        "Each section is a horizontal, light-mode strip: one card per class,",
        "left to right, with its swatch, class id/name and exact hex value.",
        "",
        "| File | Content |",
        "|---|---|",
        f"| `{config.FLOORPLAN_IMAGE_NAME}` | Floorplan classes ({len(floorplan)}) |",
        (
            f"| `{config.ARRANGEMENT_IMAGE_NAME}` | "
            f"Arrangement/furniture classes ({len(arrangement)}) |"
        ),
        f"| `{config.COMBINED_IMAGE_NAME}` | Both sections in one image |",
        f"| `{config.JSON_NAME}` | Both tables as machine-readable JSON |",
        "",
        "Class id 0 is always background (black). Colors are the exact hex",
        "values used by the mask renderers; a class whose color repeats another",
        "(for example `sofa`, `couch` and `chair`) keeps its own separate class id,",
        "so read the name column rather than the color alone.",
        "",
        "Sources:",
        "",
        f"- `{Path(str(config.COLOR_INDEX_DIR)) / str(config.FLOORPLAN_COLOR_INDEX_FILE)}`",
        f"- `{Path(str(config.COLOR_INDEX_DIR)) / str(config.ARRANGEMENT_COLOR_INDEX_FILE)}`",
        "",
    ]
    readme_path: Path = output_root / str(config.README_NAME)
    readme_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return readme_path


def resolve_output_root(path: str) -> Path:
    """Resolve an output directory against the repository root.

    Args:
        path (str): Absolute path or repository-root relative path.

    Returns:
        Path: Absolute output directory, created if missing.
    """
    candidate: Path = Path(path).expanduser()
    resolved: Path = candidate if candidate.is_absolute() else ROOT_DIR / candidate
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line options that override the TOML defaults.

    Args:
        argv (list[str] | None): Argument list, defaulting to ``sys.argv[1:]``.

    Returns:
        argparse.Namespace: Parsed options.
    """
    parser = argparse.ArgumentParser(
        description="Render a legend for the floorplan and arrangement mask colors."
    )
    parser.add_argument(
        "--mode",
        choices=["debug", "prod"],
        default="debug",
        help="Runtime configuration mode (debug or prod).",
    )
    parser.add_argument(
        "--index-dir",
        type=str,
        default=None,
        help="Directory holding the two color index JSON files.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Destination for the legends and merged JSON.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Build every color-index artifact from the command line.

    Args:
        argv (list[str] | None): Argument list, defaulting to ``sys.argv[1:]``.

    Returns:
        int: Process exit code, ``0`` on success.
    """
    script_dir: Path = Path(__file__).resolve().parent
    args: argparse.Namespace = parse_arguments(argv)
    config: SimpleNamespace = load_config(args.mode, str(script_dir / SCRIPT_TOML))
    index_dir: Path = Path(
        args.index_dir or str(config.COLOR_INDEX_DIR)
    ).expanduser()
    floorplan_path: Path = index_dir / str(config.FLOORPLAN_COLOR_INDEX_FILE)
    arrangement_path: Path = index_dir / str(config.ARRANGEMENT_COLOR_INDEX_FILE)
    output_root: Path = resolve_output_root(
        str(args.output_dir or config.OUTPUT_DIRECTORY)
    )
    log: SimpleNamespace = setup_logger(config)

    style: IndexStyle = IndexStyle.from_config(config)
    floorplan: list[IndexEntry] = load_entries(floorplan_path)
    arrangement: list[IndexEntry] = load_entries(arrangement_path)
    log.info(
        f"Loaded {len(floorplan)} floorplan classes and "
        f"{len(arrangement)} arrangement classes."
    )

    floorplan_image: Image.Image = draw_legend(
        floorplan,
        title="Floorplan mask colors",
        subtitle=str(floorplan_path.name),
        style=style,
    )
    arrangement_image: Image.Image = draw_legend(
        arrangement,
        title="Arrangement (furniture) mask colors",
        subtitle=str(arrangement_path.name),
        style=style,
    )
    combined: Image.Image = combine_legends(floorplan_image, arrangement_image, style)

    floorplan_image.save(str(output_root / str(config.FLOORPLAN_IMAGE_NAME)))
    arrangement_image.save(str(output_root / str(config.ARRANGEMENT_IMAGE_NAME)))
    combined.save(str(output_root / str(config.COMBINED_IMAGE_NAME)))
    payload: dict[str, object] = build_json_payload(
        floorplan_path, arrangement_path, floorplan, arrangement
    )
    (output_root / str(config.JSON_NAME)).write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    write_readme(output_root, config, floorplan, arrangement)
    log.info(f"Color index written to {output_root}.")
    print(f"Color index written to {output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
