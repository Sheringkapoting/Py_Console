# face_sorter.py

Sorts images into per-person subfolders using **InsightFace**
(SCRFD detection + ArcFace embedding, `buffalo_l` model pack). Fully
interactive if flags are omitted.

**Install**: heavier ML stack, **not** in `requirements.txt` — see
`docs/dependencies.md` Tier 2. `pip install insightface onnxruntime
opencv-python pillow tqdm rich psutil` (or `onnxruntime-directml` instead
of `onnxruntime` for DirectML/Intel Arc GPU acceleration).

## Flags

| Flag | Default | Description |
|---|---|---|
| `--src` | prompted | Folder of images to sort |
| `--faces` | prompted | Folder of reference photos, one per person |
| `--models` | this repo's own directory | Root folder containing `models\buffalo_l\` |
| `--report` | `<src>\face_sort_report.html` | HTML report path |
| `--threshold` | tool default | Cosine similarity threshold (0–1) |
| `--recursive` | off | Scan subfolders of source |
| `--execute` | off | Move matched files (default: dry run) |

## Examples

```bash
python face_sorter.py                                       # fully interactive
python face_sorter.py --execute                              # skip dry-run prompt
python face_sorter.py --src "D:\Photos" --faces "D:\Refs"
python face_sorter.py --threshold 0.45 --recursive
```

## Notes

- `--models` defaults to this repo's own directory (resolved from the
  script's location), so `models\buffalo_l\` is found automatically when
  running from a checkout of this repo — no flag needed in the common
  case. Only pass `--models` if your model files live somewhere else.
  (Previously defaulted to a personal path on the original author's
  machine; fixed — see `models/CLAUDE.md`.)
- ESC aborts cleanly mid-scan.
