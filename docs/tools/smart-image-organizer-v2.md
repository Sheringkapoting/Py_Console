# smart_image_organizer_v2.py

Clusters images by **combined signals** — face embeddings (InsightFace),
CLIP scene/theme embeddings, background features, and time proximity —
via agglomerative clustering, then renames each cluster sequentially so
related images sort together by filename.

**Install**: heavier ML stack, **not** in `requirements.txt` — see
`docs/dependencies.md` Tier 2. Needs `insightface`, `onnxruntime` (or
`onnxruntime-directml`), `torch`, `clip`, `opencv-python`, `scikit-learn`
— pulls in as much as `face_sorter.py` and `image_tagger.py` combined.

## Flags

| Flag | Default | Description |
|---|---|---|
| `--src` | prompted | Source image folder |
| `--recursive` | off | Recurse into subfolders |
| `--models-root` | this repo's own directory | Parent of `models/buffalo_l/` |
| `--prefix` | `series` | Rename prefix |
| `--cluster-threshold` | `0.48` | Agglomerative distance threshold (0–1) |
| `--time-half-life-hours` | `72.0` | Time-proximity signal half-life |
| `--weight-face` | `0.40` | Face-embedding signal weight |
| `--weight-clip` | `0.28` | CLIP scene/theme signal weight |
| `--weight-bg` | `0.17` | Background-feature signal weight |
| `--weight-time` | `0.15` | Time-proximity signal weight |
| `--no-continue-series` | off | Restart numbering from 001 |
| `--dry-run` | off | Preview renames without touching files |
| `--force` | off | Re-cluster files that already match the prefix pattern |

The four `--weight-*` flags should sum to ~1.0 if changed together.

## Examples

```bash
python smart_image_organizer_v2.py --src "D:\Photos" --dry-run
python smart_image_organizer_v2.py --src "D:\Photos" --prefix trip --cluster-threshold 0.4
```

## Notes

- `--models-root` defaults to this repo's own directory, same fix and
  same reasoning as `face_sorter.py` — see `models/CLAUDE.md`. No flag
  needed in the common case of running from a checkout of this repo.
- `--dry-run` here is a real, explicit flag (unlike some other tools in
  this repo where dry-run is only the *absence* of `--execute`) — see
  `docs/conventions.md`.
