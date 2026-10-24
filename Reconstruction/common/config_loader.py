"""TOML loading for the Reconstruction package.

Two kinds of TOML files are used:

* **Run configs** (``config.toml``) with ``[default]``, ``[debug]`` and
  ``[prod]`` tables. The selected mode's table overrides ``[default]``,
  mirroring the convention used elsewhere in this repository.
* **Parameter files** (``params.toml``) with arbitrary nested tables. They are
  converted to nested ``SimpleNamespace`` objects so values read as
  ``params.canvas.SIZE``.

Relative paths inside configs are resolved against the repository root, so
scripts behave the same regardless of the working directory they run from.
"""

import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

# tomllib is in the standard library from Python 3.11; the project conda env
# currently ships 3.10, where the API-compatible ``tomli`` backport is used.
if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib

# Repository root: Reconstruction/common/config_loader.py -> parents[2].
REPO_ROOT: Path = Path(__file__).resolve().parents[2]


def to_namespace(value: Any) -> Any:
    """Recursively convert dicts (TOML tables) into ``SimpleNamespace`` objects.

    Args:
        value (Any): A TOML value; tables become namespaces, lists are
            converted element-wise and scalars are returned unchanged.

    Returns:
        Any: The converted value.
    """
    if isinstance(value, dict):
        return SimpleNamespace(**{key: to_namespace(item) for key, item in value.items()})
    if isinstance(value, list):
        return [to_namespace(item) for item in value]
    return value


def read_toml(path: Path) -> dict[str, Any]:
    """Read a TOML file into a plain dictionary.

    Args:
        path (Path): Path to the TOML file.

    Returns:
        dict[str, Any]: Parsed TOML content.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
    """
    if not path.is_file():
        raise FileNotFoundError(f"TOML file not found: {path}")
    with open(path, "rb") as handle:
        return tomllib.load(handle)


def load_run_config(path: Path, mode: str) -> SimpleNamespace:
    """Load a run config, applying the ``[mode]`` table over ``[default]``.

    Args:
        path (Path): Path to the run config TOML.
        mode (str): Run mode, the name of the override table (e.g. "debug").

    Returns:
        SimpleNamespace: Flat namespace of config keys plus ``MODE``.

    Raises:
        KeyError: If the file has no ``[default]`` table or no ``[mode]`` table.
    """
    data: dict[str, Any] = read_toml(path)
    if "default" not in data:
        raise KeyError(f"{path} has no [default] table")
    if mode not in data:
        raise KeyError(f"{path} has no [{mode}] table; available: {sorted(data)}")
    merged: dict[str, Any] = {**data["default"], **data[mode], "MODE": mode}
    return SimpleNamespace(**merged)


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge ``override`` into a copy of ``base``.

    Args:
        base (dict[str, Any]): Base tables.
        override (dict[str, Any]): Tables whose values replace base values.

    Returns:
        dict[str, Any]: Merged tables.

    Raises:
        KeyError: If the override names a key the base does not have (typo guard).
    """
    merged: dict[str, Any] = dict(base)
    for key, value in override.items():
        if key not in base:
            raise KeyError(f"Override key '{key}' does not exist in the base parameters")
        merged[key] = _deep_merge(base[key], value) if isinstance(value, dict) else value
    return merged


def load_params(path: Path, mode: str | None = None) -> SimpleNamespace:
    """Load a nested parameter TOML as nested namespaces.

    A file may contain an ``[overrides.<mode>...]`` tree; with ``mode`` given,
    it is deep-merged over the other tables (e.g. a tiny debug model).

    Args:
        path (Path): Path to the parameter TOML.
        mode (str | None): Override set to apply (None = base values).

    Returns:
        SimpleNamespace: Nested namespace mirroring the TOML tables.

    Raises:
        KeyError: If ``mode`` is given but has no override table.
    """
    data: dict[str, Any] = read_toml(path)
    overrides: dict[str, Any] = data.pop("overrides", {})
    if mode is not None:
        if mode not in overrides:
            raise KeyError(f"{path} has no [overrides.{mode}] table; available: {sorted(overrides)}")
        data = _deep_merge(data, overrides[mode])
    return to_namespace(data)


def resolve_path(path_str: str) -> Path:
    """Resolve a config path; relative paths are taken from the repository root.

    Args:
        path_str (str): Absolute path, or path relative to the repo root.

    Returns:
        Path: Absolute resolved path.
    """
    path: Path = Path(path_str).expanduser()
    return path if path.is_absolute() else (REPO_ROOT / path).resolve()
