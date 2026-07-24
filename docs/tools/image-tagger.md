# image_tagger.py

Semantic image tagging via **CLIP** (zero-shot scene/theme
classification) + **MediaPipe Pose** (posture labels — standing,
sitting, crouching, etc.). Writes a structured `tags.json` with every
label, confidence, runner-up alternatives, and a suggested rename stem;
can also group/rename directly.

**Install**: heavier ML stack, **not** in `requirements.txt` — see
`docs/dependencies.md` Tier 2. CPU-only PyTorch recommended unless you
have a GPU: `pip install torch --index-url
https://download.pytorch.org/whl/cpu`, then `pip install transformers
mediapipe opencv-python pillow rich tqdm`. The CLIP model (~350 MB,
ViT-B/32 default) downloads automatically on first run.

## Flags

| Flag | Default | Description |
|---|---|---|
| `--src` | prompted | Source folder |
| `--recursive` | off | Scan subfolders |
| `--group` | off | Sort into `<scene>/<theme>/` subfolders |
| `--rename` | off | Rename to `<scene>_<theme>_<posture>_NNNN.ext` |
| `--copy` | off | Copy instead of move when grouping |
| `--execute` | off | Apply group/rename (default: tag-only, no file changes) |
| `--threshold` | tool default | Min CLIP softmax score to accept a label |
| `--clip-model` | tool default | HuggingFace CLIP model ID |
| `--save-embeddings` | off | Include 512-dim CLIP vectors in `tags.json` |
| `--tags-path` | `<src>/tags.json` | Output path for the tags file |

## Examples

```bash
python image_tagger.py --src "D:\Photos"                       # tag only, writes tags.json
python image_tagger.py --src "D:\Photos" --group --execute      # sort into scene/theme folders
python image_tagger.py --src "D:\Photos" --rename --execute
```

## Notes

- `pose_landmarker.task` (repo root, LFS-tracked) is MediaPipe's pose
  model — see `models/CLAUDE.md` for the same no-touch/no-`Read` rules
  that apply to `models/`.
- `--execute` gates *all* file changes here (both `--group` and
  `--rename`) — without it, the tool only writes `tags.json`.
