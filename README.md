# Metanoia
Architectural Floor Plan Missing Components Generation

## Utilities

### Mask color index

`index_maker.py` turns the dataset's `floorplan_color_index.json` and
`furniture_color_index.json` into readable horizontal, light-mode legend
images and a merged index:

```bash
python index_maker.py
```

Outputs go to `outputs/color_index/`:

| File | Content |
|---|---|
| `index_floorplan.png` | Floorplan classes (rooms, walls, doors, ...) |
| `index_arrangement.png` | Arrangement/furniture classes (sofa, tv, dining set, ...) |
| `index_combined.png` | Both legends in one image |
| `color_index.json` | Both class tables, machine-readable |

Paths, colors and pixel sizes are configured in `index_maker.toml`.
