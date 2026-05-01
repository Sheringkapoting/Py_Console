#!/usr/bin/env python3
"""
image_tagger.py  ─  CLIP + MediaPipe semantic image tagger
===========================================================
Analyses images using two complementary models:

  • CLIP (ViT-B/32 by default)   — zero-shot scene, background, and visual
                                   theme classification via natural-language
                                   label matching
  • MediaPipe Pose               — body-landmark detection → posture label
                                   (standing, sitting, crouching, etc.)

Outputs a structured  tags.json  file that records every label, its
confidence score, runner-up alternatives, a group key, and a suggested
rename stem.  The same JSON can drive grouping, renaming, or filtering
in a second pass — or with this script's own --group / --rename actions.

Requirements (install once):
    # CPU-only PyTorch (~250 MB, recommended unless you have a GPU):
    pip install torch --index-url https://download.pytorch.org/whl/cpu

    # Everything else:
    pip install transformers mediapipe opencv-python pillow rich tqdm

    CLIP model download (~350 MB for ViT-B/32) happens automatically on
    the first run and is cached to  ~/.cache/huggingface/.

Usage:
    python image_tagger.py                          # fully interactive
    python image_tagger.py --src "D:\\Photos"       # skip source prompt
    python image_tagger.py --src "D:\\Photos" --group --execute
    python image_tagger.py --src "D:\\Photos" --rename --execute
    python image_tagger.py --recursive --save-embeddings
    python image_tagger.py --clip-model openai/clip-vit-large-patch14
    python image_tagger.py --help

tags.json structure (one entry per image):
    {
      "filename": "IMG_001.jpg",
      "path":     "C:\\Photos\\IMG_001.jpg",
      "scene":  { "label": "beach or seaside", "score": 0.82,
                  "accepted": true,
                  "top3": [["beach or seaside", 0.82], ...] },
      "theme":  { "label": "beach holiday or seaside vacation",
                  "score": 0.74, "accepted": true, "top3": [...] },
      "posture":{ "detected": true, "label": "standing_arms_raised",
                  "score": 0.82, "person_count": 1 },
      "group_key":      "beach__beach_holiday",
      "suggested_name": "beach_beach_holiday_standing_arms_raised",
      "clip_embedding": null          # or [512 floats] with --save-embeddings
    }
"""

from __future__ import annotations

# ══════════════════════════════════════════════════════════════════════════════
# LABEL SETS  —  edit freely; more descriptive phrases work better than single
#                words because CLIP was trained on natural-language captions.
# ══════════════════════════════════════════════════════════════════════════════

SCENE_LABELS: list[str] = [
    "beach or seaside",
    "mountain or hillside",
    "forest or woodland",
    "city street or urban environment",
    "indoor room or home interior",
    "garden or public park",
    "desert or arid landscape",
    "snowy or winter scenery",
    "sports venue or stadium",
    "restaurant or cafe",
    "swimming pool or water park",
    "office or workplace",
    "temple or religious building",
    "shopping area or market",
]

THEME_LABELS: list[str] = [
    "wedding ceremony or wedding celebration",
    "birthday party or celebration event",
    "beach holiday or seaside vacation",
    "outdoor adventure or nature trek",
    "family gathering or reunion",
    "sports or fitness activity",
    "fashion shoot or portrait session",
    "food or dining experience",
    "travel and sightseeing",
    "casual everyday moment",
    "formal event or gala dinner",
    "cultural festival or traditional ceremony",
    "graduation or academic celebration",
    "live music or performance event",
]

# ── Script-level defaults (all overridable via CLI or interactive prompt) ─────
_DEFAULT_CLIP_MODEL    = "openai/clip-vit-base-patch32"
_DEFAULT_THRESHOLD     = 0.20   # minimum CLIP softmax score to mark a label "accepted"
_DEFAULT_POSE_CONF     = 0.55   # MediaPipe minimum detection confidence
_IMAGE_EXTS            = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
# ─────────────────────────────────────────────────────────────────────────────

import argparse
import json
import os
import re
import shutil
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Optional


# ══════════════════════════════════════════════════════════════════════════════
# Dependency check  (runs before any heavy import)
# ══════════════════════════════════════════════════════════════════════════════

def _check_deps() -> None:
    pairs = [
        ("torch",          "torch"),
        ("transformers",   "transformers"),
        ("mediapipe",      "mediapipe"),
        ("opencv-python",  "cv2"),
        ("pillow",         "PIL"),
        ("rich",           "rich"),
        ("tqdm",           "tqdm"),
    ]
    missing = [pkg for pkg, mod in pairs if not _importable(mod)]
    if not missing:
        return

    print("\n  ✗  Missing packages.  Install them with:\n")
    if "torch" in missing:
        print("     pip install torch --index-url https://download.pytorch.org/whl/cpu")
        missing.remove("torch")
    if missing:
        print(f"     pip install {' '.join(missing)}")
    print()
    sys.exit(1)


def _importable(name: str) -> bool:
    try:
        __import__(name)
        return True
    except ImportError:
        return False


_check_deps()

# ── Imports (all available after dep-check) ───────────────────────────────────

import os
os.environ["GLOG_minloglevel"] = "2"   # suppress MediaPipe / glog verbose warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import cv2
import numpy as np
import torch
from PIL import Image, ImageOps
from transformers import CLIPModel, CLIPProcessor
import mediapipe as mp
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt, Confirm
from rich.rule import Rule
from rich.table import Table
from rich.text import Text
from rich.progress import (
    BarColumn, Progress, SpinnerColumn,
    TaskProgressColumn, TextColumn,
    TimeElapsedColumn, TimeRemainingColumn,
)

console = Console()


# ══════════════════════════════════════════════════════════════════════════════
# CLIP  —  zero-shot scene + theme classification
# ══════════════════════════════════════════════════════════════════════════════

class CLIPClassifier:
    """
    Wraps a HuggingFace CLIP model for zero-shot image classification.

    Each call to  classify()  computes a softmax over a list of text labels
    and returns (best_label, best_score, top3).  The model and processor are
    loaded once and reused for every image.
    """

    def __init__(self, model_name: str = _DEFAULT_CLIP_MODEL) -> None:
        self.model_name = model_name
        with console.status(f"[cyan]Loading CLIP  ({model_name}) …[/cyan]"):
            self._model     = CLIPModel.from_pretrained(model_name)
            self._processor = CLIPProcessor.from_pretrained(model_name)
            self._model.eval()
        console.print(f"  [green]✓[/green]  CLIP ready  ({model_name})")

    # ------------------------------------------------------------------
    def classify(
        self,
        image: Image.Image,
        labels: list[str],
        caption_prefix: str = "a photo of ",
    ) -> tuple[str, float, list[tuple[str, float]]]:
        """
        Zero-shot classify *image* against *labels*.

        Returns:
            best_label  — the highest-scoring label string
            best_score  — its softmax probability (0–1)
            top3        — [(label, score), …] for the top 3 candidates
        """
        texts  = [caption_prefix + lbl for lbl in labels]
        inputs = self._processor(
            text=texts, images=image, return_tensors="pt", padding=True
        )
        with torch.no_grad():
            probs = self._model(**inputs).logits_per_image.softmax(dim=-1)[0].tolist()

        ranked = sorted(zip(labels, probs), key=lambda x: -x[1])
        top3   = [(lbl, round(sc, 4)) for lbl, sc in ranked[:3]]
        return ranked[0][0], round(ranked[0][1], 4), top3

    # ------------------------------------------------------------------
    def embedding(self, image: Image.Image) -> list[float]:
        """Return the L2-normalised CLIP image embedding as a plain Python list."""
        inputs = self._processor(images=image, return_tensors="pt")
        with torch.no_grad():
            feat = self._model.get_image_features(**inputs)
            feat = feat / feat.norm(dim=-1, keepdim=True)
        return feat[0].tolist()


# ══════════════════════════════════════════════════════════════════════════════
# MediaPipe  —  body posture classification
# ══════════════════════════════════════════════════════════════════════════════

# Landmark indices referenced in posture logic (from MediaPipe's 33-point model)
_LM = dict(
    l_shoulder=11, r_shoulder=12,
    l_elbow=13,    r_elbow=14,
    l_wrist=15,    r_wrist=16,
    l_hip=23,      r_hip=24,
    l_knee=25,     r_knee=26,
    l_ankle=27,    r_ankle=28,
)


# ── MediaPipe Tasks API model download ───────────────────────────────────────
# MediaPipe >= 0.10.14 removed mp.solutions entirely.
# The new Tasks API requires a small .task model file downloaded once.

_POSE_MODEL_URL  = (
    "https://storage.googleapis.com/mediapipe-models/"
    "pose_landmarker/pose_landmarker_full/float16/latest/"
    "pose_landmarker_full.task"
)
_POSE_MODEL_PATH = Path.home() / ".cache" / "mediapipe_tasks" / "pose_landmarker_full.task"


def _ensure_pose_model() -> Optional[Path]:
    """
    Return the path to the MediaPipe Tasks pose model, downloading it
    (~6 MB) on first use if it is not already cached.
    """
    if _POSE_MODEL_PATH.exists():
        return _POSE_MODEL_PATH
    try:
        import urllib.request
        _POSE_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        console.print("  [cyan]Downloading MediaPipe pose model (~6 MB, one-time) …[/cyan]")
        urllib.request.urlretrieve(_POSE_MODEL_URL, str(_POSE_MODEL_PATH))
        console.print("  [green]✓[/green]  Pose model saved to cache")
        return _POSE_MODEL_PATH
    except Exception as exc:
        console.print(f"  [yellow]⚠  Could not download pose model: {exc}[/yellow]")
        return None


class PoseEstimator:
    """
    Wraps MediaPipe Pose for single-image body-posture classification.

    Automatically selects the correct API for the installed MediaPipe version:

    • New Tasks API  (mediapipe >= 0.10.14) — mp.solutions no longer exists;
      uses mediapipe.tasks.vision.PoseLandmarker with a downloaded .task file.

    • Legacy Solutions API (mediapipe < 0.10.14) — mp.solutions.pose.Pose,
      available in older installs as a fallback.

    Both paths produce the same 33-landmark sequence fed to _classify_posture().
    """

    def __init__(self, min_confidence: float = _DEFAULT_POSE_CONF) -> None:
        self._mode       : str  = ""
        self._landmarker        = None   # Tasks API
        self._legacy_pose       = None   # Solutions API
        self._mp_vision         = None
        self._min_conf   = min_confidence

        # ── Try new Tasks API first (mediapipe >= 0.10.14) ───────────────────
        try:
            from mediapipe.tasks import python as _mp_python
            from mediapipe.tasks.python import vision as _mp_vision

            model_path = _ensure_pose_model()
            if model_path is None:
                raise RuntimeError("pose model unavailable")

            base_opts = _mp_python.BaseOptions(model_asset_path=str(model_path))
            opts = _mp_vision.PoseLandmarkerOptions(
                base_options                  = base_opts,
                running_mode                  = _mp_vision.RunningMode.IMAGE,
                num_poses                     = 1,
                min_pose_detection_confidence = min_confidence,
                min_pose_presence_confidence  = min_confidence,
                min_tracking_confidence       = min_confidence,
            )
            self._landmarker = _mp_vision.PoseLandmarker.create_from_options(opts)
            self._mp_vision  = _mp_vision
            self._mode       = "tasks"
            console.print("  [green]✓[/green]  MediaPipe Pose ready  [dim](Tasks API)[/dim]")
            return
        except Exception:
            pass

        # ── Fallback: legacy Solutions API (mediapipe < 0.10.14) ─────────────
        try:
            self._legacy_pose = mp.solutions.pose.Pose(
                static_image_mode        = True,
                model_complexity         = 1,
                min_detection_confidence = min_confidence,
            )
            self._mode = "solutions"
            console.print("  [green]✓[/green]  MediaPipe Pose ready  [dim](Solutions API)[/dim]")
            return
        except AttributeError:
            pass

        console.print(
            "  [red]✗  MediaPipe Pose could not be initialised.[/red]\n"
            "     Posture detection will be skipped for all images.\n"
            "     To fix: ensure internet access for the model download,\n"
            "     or run:  pip install mediapipe==0.10.11"
        )
        self._mode = "disabled"

    # ------------------------------------------------------------------
    def analyse(self, img_path: Path) -> dict:
        """
        Run pose estimation on one image file.
        Returns a dict with keys: detected, label, score, person_count.
        """
        if self._mode == "disabled":
            return _pose_none()

        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            return _pose_none()

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # ── Tasks API path ───────────────────────────────────────────────────
        if self._mode == "tasks":
            try:
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
                result   = self._landmarker.detect(mp_image)
            except Exception:
                return _pose_none()

            if not result.pose_landmarks:
                return _pose_none()

            # Tasks API: pose_landmarks is List[List[NormalizedLandmark]]
            # [0] = first detected person, each element has .x .y .visibility
            lm = result.pose_landmarks[0]

        # ── Solutions API path ───────────────────────────────────────────────
        else:
            results = self._legacy_pose.process(img_rgb)
            if not results.pose_landmarks:
                return _pose_none()
            lm = _extract_landmarks(results.pose_landmarks)
            if lm is None:
                return _pose_none()

        label, score = _classify_posture(lm)
        return {"detected": True, "label": label, "score": round(score, 3), "person_count": 1}


def _pose_none() -> dict:
    return {"detected": False, "label": None, "score": None, "person_count": 0}


def _extract_landmarks(pose_landmarks):
    """
    Normalise the MediaPipe landmark result across all API versions.

    MediaPipe has shipped three different shapes for pose_landmarks depending
    on version and API:

    v1  (solutions API ≤ 0.9):
        NormalizedLandmarkList  — has .landmark attribute
        → return .landmark  (a repeated-field iterable of NormalizedLandmark)

    v2  (solutions API ≥ 0.10 in some builds):
        List[NormalizedLandmarkList]  — outer list per detected person
        → return [0].landmark

    v3  (Tasks API / some 0.10 builds):
        List[List[NormalizedLandmark]]  — outer list per person, inner = landmarks
        → return [0]  directly

    All three produce a sequence of 33 objects with .x / .y / .visibility.
    """
    if pose_landmarks is None:
        return None

    # v1: single NormalizedLandmarkList with .landmark attribute
    if hasattr(pose_landmarks, "landmark"):
        return pose_landmarks.landmark

    # v2 / v3: outer list (one entry per detected person)
    if isinstance(pose_landmarks, (list, tuple)):
        if len(pose_landmarks) == 0:
            return None
        first = pose_landmarks[0]
        if hasattr(first, "landmark"):       # v2: NormalizedLandmarkList
            return first.landmark
        if isinstance(first, (list, tuple)): # v3: bare list of landmarks
            return first

    return None


# ── Posture rules ─────────────────────────────────────────────────────────────
#
# MediaPipe normalised coords:   x ∈ [0,1]  (0 = left edge)
#                                y ∈ [0,1]  (0 = top,  1 = bottom — y grows DOWN)
#
# Detection priority (most specific first):
#   lying_down → crouching → sitting → standing_arms_raised → standing

def _y(lm, k: str) -> float:
    return lm[_LM[k]].y

def _vis(lm, k: str) -> bool:
    return lm[_LM[k]].visibility > 0.40

def _mid_y(lm, ka: str, kb: str) -> float:
    """Average y of two landmarks (visibility-weighted)."""
    va, vb = _vis(lm, ka), _vis(lm, kb)
    if va and vb:
        return (_y(lm, ka) + _y(lm, kb)) / 2
    return _y(lm, ka) if va else _y(lm, kb)


def _classify_posture(lm) -> tuple[str, float]:
    sh_y    = _mid_y(lm, "l_shoulder", "r_shoulder")
    hip_y   = _mid_y(lm, "l_hip",      "r_hip")
    torso_h = hip_y - sh_y   # > 0 when shoulders are above hips (upright)

    have_knees  = _vis(lm, "l_knee")  or _vis(lm, "r_knee")
    have_ankles = _vis(lm, "l_ankle") or _vis(lm, "r_ankle")
    have_wrists = _vis(lm, "l_wrist") or _vis(lm, "r_wrist")

    kn_y = _mid_y(lm, "l_knee",  "r_knee")  if have_knees  else None
    an_y = _mid_y(lm, "l_ankle", "r_ankle") if have_ankles else None

    # ── Lying down: shoulders and hips at roughly the same vertical level
    if abs(torso_h) < 0.08:
        return "lying_down", 0.80

    # ── Crouching: knees barely below hips AND ankles barely below knees
    if have_knees and have_ankles and kn_y and an_y and torso_h > 0.05:
        if (kn_y - hip_y) < 0.12 and (an_y - kn_y) < 0.12:
            return "crouching", 0.75

    # ── Sitting: hips and knees at nearly the same height
    if have_knees and kn_y and abs(kn_y - hip_y) < 0.10 and torso_h > 0.05:
        return "sitting", 0.78

    # ── Arms raised: one or both wrists clearly above the shoulders
    if have_wrists:
        left_up  = _vis(lm, "l_wrist") and (_y(lm, "l_wrist") < sh_y - 0.06)
        right_up = _vis(lm, "r_wrist") and (_y(lm, "r_wrist") < sh_y - 0.06)
        if left_up or right_up:
            return "standing_arms_raised", 0.82

    return "standing", 0.88


# ══════════════════════════════════════════════════════════════════════════════
# Tag generation  —  one record per image
# ══════════════════════════════════════════════════════════════════════════════

def _slugify(text: str) -> str:
    """
    Convert a verbose CLIP label to a compact filename-safe slug.

    Examples:
        "beach or seaside"                  →  "beach"
        "wedding ceremony or celebration"   →  "wedding_ceremony"
        "indoor room or home interior"      →  "indoor_room"
    """
    # Keep only the phrase before the first " or "
    text = text.split(" or ")[0].strip()
    # Lower-case and replace non-alphanumeric runs with underscore
    text = re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")
    return text


def _group_key(scene_label: str, theme_label: str) -> str:
    s, t = _slugify(scene_label), _slugify(theme_label)
    return f"{s}__{t}" if s != t else s


def tag_image(
    img_path: Path,
    clip: CLIPClassifier,
    pose: PoseEstimator,
    threshold: float,
    save_embedding: bool,
) -> dict:
    """
    Run the full CLIP + MediaPipe analysis pipeline on a single image.

    Returns a flat dict that is serialised directly into tags.json.
    On read error, returns a minimal record with an "error" key so the
    caller can always produce a complete JSON (no silently skipped files).
    """
    try:
        pil_img = Image.open(img_path)
        pil_img = ImageOps.exif_transpose(pil_img).convert("RGB")
    except Exception as exc:
        return {"filename": img_path.name, "path": str(img_path), "error": str(exc)}

    scene_lbl, scene_sc, scene_top3 = clip.classify(pil_img, SCENE_LABELS)
    theme_lbl, theme_sc, theme_top3 = clip.classify(pil_img, THEME_LABELS)
    posture                          = pose.analyse(img_path)
    emb                              = clip.embedding(pil_img) if save_embedding else None

    posture_slug = posture["label"] if posture["label"] else "no_pose"
    group        = _group_key(scene_lbl, theme_lbl)

    return {
        "filename": img_path.name,
        "path":     str(img_path),
        "scene": {
            "label":    scene_lbl,
            "score":    scene_sc,
            "accepted": scene_sc >= threshold,
            "top3":     scene_top3,
        },
        "theme": {
            "label":    theme_lbl,
            "score":    theme_sc,
            "accepted": theme_sc >= threshold,
            "top3":     theme_top3,
        },
        "posture":        posture,
        "group_key":      group,
        "suggested_name": f"{_slugify(scene_lbl)}_{_slugify(theme_lbl)}_{posture_slug}",
        "clip_embedding": emb,
    }


# ══════════════════════════════════════════════════════════════════════════════
# Output actions  —  group and/or rename
# ══════════════════════════════════════════════════════════════════════════════

def action_group(
    tagged: list[dict],
    src: Path,
    copy_mode: bool,
) -> tuple[int, int]:
    """
    Move (or copy) each image into  <src>/<scene_slug>/<theme_slug>/<filename>.

    Uses "accepted" flag so images whose CLIP score fell below the threshold
    land in an  _unclassified/  folder rather than being mis-sorted.

    Returns (files_moved, errors).
    """
    collision_counter: dict[str, int] = defaultdict(int)
    moved = errors = 0

    for rec in tagged:
        if "error" in rec:
            continue

        if rec["scene"]["accepted"] and rec["theme"]["accepted"]:
            sub = Path(_slugify(rec["scene"]["label"])) / _slugify(rec["theme"]["label"])
        else:
            sub = Path("_unclassified")

        dest_dir = src / sub
        dest_dir.mkdir(parents=True, exist_ok=True)

        src_path = Path(rec["path"])
        dest     = dest_dir / src_path.name
        if dest.exists():
            collision_counter[str(dest_dir)] += 1
            n    = collision_counter[str(dest_dir)]
            dest = dest_dir / (src_path.stem + f"_{n}" + src_path.suffix)

        try:
            shutil.copy2(str(src_path), str(dest)) if copy_mode else shutil.move(str(src_path), str(dest))
            rec["moved_to"] = str(dest)
            moved += 1
        except Exception as exc:
            console.print(f"  [red]ERR[/red]  {src_path.name}: {exc}")
            errors += 1

    return moved, errors


def action_rename(tagged: list[dict]) -> tuple[int, int]:
    """
    Rename each image in place using the suggested_name stem:
        <scene>_<theme>_<posture>_<NNNN><original_ext>

    Returns (renamed, errors).
    """
    counter: dict[str, int] = defaultdict(int)
    renamed = errors = 0

    for rec in tagged:
        if "error" in rec:
            continue

        base     = rec["suggested_name"]
        counter[base] += 1
        ext      = Path(rec["path"]).suffix.lower()
        new_name = f"{base}_{counter[base]:04d}{ext}"
        old      = Path(rec["path"])
        new      = old.parent / new_name

        if new.exists():
            console.print(f"  [yellow]SKIP[/yellow]  {new_name} already exists")
            continue
        try:
            old.rename(new)
            rec["renamed_to"] = str(new)
            renamed += 1
        except Exception as exc:
            console.print(f"  [red]ERR[/red]  {old.name}: {exc}")
            errors += 1

    return renamed, errors


# ══════════════════════════════════════════════════════════════════════════════
# Image collection
# ══════════════════════════════════════════════════════════════════════════════

def collect_images(src: Path, recursive: bool) -> list[Path]:
    pattern = "**/*" if recursive else "*"
    return sorted(
        p for p in src.glob(pattern)
        if p.is_file() and p.suffix.lower() in _IMAGE_EXTS
    )


# ══════════════════════════════════════════════════════════════════════════════
# Console summary
# ══════════════════════════════════════════════════════════════════════════════

def print_summary(tagged: list[dict]) -> None:
    by_group: dict[str, list[dict]] = defaultdict(list)
    for rec in tagged:
        if "error" not in rec:
            by_group[rec["group_key"]].append(rec)

    table = Table(
        title        = "Tagging summary  (by scene + theme group)",
        show_lines   = False,
        header_style = "bold cyan",
        min_width    = 72,
    )
    table.add_column("Group",         style="green")
    table.add_column("Images",        justify="right", style="bold white")
    table.add_column("Scene score",   justify="right", style="yellow")
    table.add_column("Theme score",   justify="right", style="yellow")
    table.add_column("Postures seen", style="dim")

    for group in sorted(by_group, key=lambda g: -len(by_group[g])):
        recs    = by_group[group]
        s_avg   = sum(r["scene"]["score"] for r in recs) / len(recs)
        t_avg   = sum(r["theme"]["score"] for r in recs) / len(recs)
        poses   = ", ".join(sorted({r["posture"]["label"] for r in recs if r["posture"]["label"]})) or "—"
        table.add_row(group, str(len(recs)), f"{s_avg:.3f}", f"{t_avg:.3f}", poses)

    console.print(table)

    errors = [r for r in tagged if "error" in r]
    if errors:
        console.print(f"  [red]✗  {len(errors)} images could not be read[/red]")


# ══════════════════════════════════════════════════════════════════════════════
# Interactive setup
# ══════════════════════════════════════════════════════════════════════════════

def _prompt_path(label: str, default: Optional[Path] = None) -> Path:
    """Keep prompting until the user supplies an existing folder path."""
    while True:
        hint = f" [dim](default: {default})[/dim]" if default else ""
        raw  = Prompt.ask(f"  [cyan]{label}[/cyan]{hint}").strip().strip('"\'')
        path = default if (raw == "" and default) else Path(raw)
        if path and path.exists():
            return path
        console.print(f"    [red]✗  Not found:[/red]  {path}")


def interactive_setup(args: argparse.Namespace) -> argparse.Namespace:
    console.print()
    console.print(Panel.fit(
        Text.assemble(
            ("Image Tagger", "bold cyan"),
            ("  ·  ", "dim"),
            ("CLIP + MediaPipe semantic analysis", "dim"),
        ),
        border_style="cyan",
    ))
    console.print("  Press [bold]Enter[/bold] to accept the shown default.\n")

    # ── Source folder ────────────────────────────────────────────────────────
    if args.src is None:
        args.src = _prompt_path("Source folder  (images to analyse)")
    else:
        console.print(f"  [cyan]Source folder[/cyan] : {args.src}")

    # ── Recursive scan ───────────────────────────────────────────────────────
    if not args._recursive_set:
        args.recursive = Confirm.ask("  Scan subfolders recursively?", default=False)

    # ── Output action ────────────────────────────────────────────────────────
    if not args._action_set:
        action = Prompt.ask(
            "  Output action",
            choices=["tag", "group", "rename", "group+rename"],
            default="tag",
        )
        args.group  = "group"  in action
        args.rename = "rename" in action

    # ── Copy vs move ─────────────────────────────────────────────────────────
    if args.group and not args._copy_set:
        args.copy = Confirm.ask(
            "  Copy files while grouping?  [dim](No = move)[/dim]",
            default=False,
        )

    # ── Execute gate ─────────────────────────────────────────────────────────
    if (args.group or args.rename) and not args.execute:
        console.print()
        args.execute = Confirm.ask(
            "  [yellow]Files will be modified. Proceed?[/yellow]",
            default=False,
        )

    # ── Tags output path ─────────────────────────────────────────────────────
    if args.tags_path is None:
        args.tags_path = Path(args.src) / "tags.json"

    # ── Confirmation panel ───────────────────────────────────────────────────
    console.print()
    console.print(Rule("Settings", style="cyan"))
    console.print(f"  Source       : [white]{args.src}[/white]")
    console.print(f"  Recursive    : [white]{'yes' if args.recursive else 'no'}[/white]")
    console.print(f"  CLIP model   : [white]{args.clip_model}[/white]")
    console.print(f"  Threshold    : [white]{args.threshold}[/white]  [dim](min score to mark label accepted)[/dim]")
    console.print(f"  Group files  : [white]{'yes (' + ('copy' if getattr(args,'copy',False) else 'move') + ')' if args.group else 'no'}[/white]")
    console.print(f"  Rename files : [white]{'yes' if args.rename else 'no'}[/white]")
    console.print(f"  Execute      : [white]{'yes' if args.execute else 'no (tag-only preview)'}[/white]")
    console.print(f"  Embeddings   : [white]{'saved in JSON' if args.save_embeddings else 'omitted'}[/white]")
    console.print(f"  Tags out     : [white]{args.tags_path}[/white]")
    console.print()

    if not Confirm.ask("  Proceed?", default=True):
        console.print("[yellow]  Aborted.[/yellow]")
        sys.exit(0)

    return args


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog             = "image_tagger",
        description      = "CLIP + MediaPipe semantic image tagger and sorter.",
        formatter_class  = argparse.RawDescriptionHelpFormatter,
        epilog = """
Output actions:
  tag           (default) Write tags.json only — no files are touched
  group         Move images into <src>/<scene>/<theme>/ subfolders
  rename        Rename images in place: <scene>_<theme>_<posture>_NNNN.jpg
  group+rename  Do both

Examples:
  python image_tagger.py
  python image_tagger.py --src "D:\\Photos" --group --execute
  python image_tagger.py --src "D:\\Photos" --rename --execute
  python image_tagger.py --src "D:\\Photos" --recursive --save-embeddings
  python image_tagger.py --clip-model openai/clip-vit-large-patch14
""",
    )
    ap.add_argument("--src",             type=Path,  default=None,
                    help="Source folder (prompted if omitted)")
    ap.add_argument("--recursive",       action="store_true", default=False,
                    help="Scan subfolders recursively")
    ap.add_argument("--group",           action="store_true", default=False,
                    help="Sort images into <scene>/<theme>/ subfolders")
    ap.add_argument("--rename",          action="store_true", default=False,
                    help="Rename images: <scene>_<theme>_<posture>_NNNN.ext")
    ap.add_argument("--copy",            action="store_true", default=False,
                    help="Copy files when grouping instead of moving")
    ap.add_argument("--execute",         action="store_true", default=False,
                    help="Apply group/rename actions (default: tag-only)")
    ap.add_argument("--threshold",       type=float, default=_DEFAULT_THRESHOLD,
                    metavar="SCORE",
                    help=f"Min CLIP softmax score to accept a label (default {_DEFAULT_THRESHOLD})")
    ap.add_argument("--clip-model",      default=_DEFAULT_CLIP_MODEL,
                    dest="clip_model",   metavar="MODEL_ID",
                    help=f"HuggingFace CLIP model ID (default: {_DEFAULT_CLIP_MODEL})")
    ap.add_argument("--save-embeddings", action="store_true", default=False,
                    dest="save_embeddings",
                    help="Include 512-dim CLIP embedding vectors in tags.json")
    ap.add_argument("--tags-path",       type=Path, default=None,
                    dest="tags_path",    metavar="PATH",
                    help="Output path for tags.json (default: <src>/tags.json)")
    return ap


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = build_parser()
    args   = parser.parse_args()

    # Flags that tell interactive_setup which questions were already answered
    args._recursive_set = "--recursive" in sys.argv
    args._action_set    = "--group" in sys.argv or "--rename" in sys.argv
    args._copy_set      = "--copy"  in sys.argv

    args = interactive_setup(args)

    src = Path(args.src)
    t0  = time.time()

    # ── Load models ──────────────────────────────────────────────────────────
    console.print()
    console.print(Rule("Loading models", style="cyan"))
    clip = CLIPClassifier(args.clip_model)
    pose = PoseEstimator()
    console.print()

    # ── Collect images ───────────────────────────────────────────────────────
    all_images = collect_images(src, args.recursive)
    if not all_images:
        console.print(f"[yellow]  ⚠  No images found in {src}[/yellow]")
        sys.exit(0)

    scope = "recursively" if args.recursive else "main folder only"
    console.print(Rule("Analysing images", style="cyan"))
    console.print(f"  [cyan]{len(all_images)}[/cyan] images  ({scope})\n")

    # ── Tagging loop ─────────────────────────────────────────────────────────
    tagged: list[dict] = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(bar_width=34),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
        refresh_per_second=4,
    ) as progress:
        task = progress.add_task("[cyan]Tagging …[/cyan]", total=len(all_images))
        for img_path in all_images:
            rec = tag_image(img_path, clip, pose, args.threshold, args.save_embeddings)
            tagged.append(rec)
            progress.advance(task)
            if "error" not in rec:
                progress.update(
                    task,
                    description=(
                        f"[cyan]Tagging[/cyan]  "
                        f"[dim]{rec['scene']['label'].split(' or ')[0][:18]}[/dim]  "
                        f"[green]{len(tagged)}[/green]"
                    ),
                )

    elapsed = time.time() - t0
    rate    = len(all_images) / elapsed if elapsed > 0 else 0
    console.print(
        f"\n  [green]✓[/green]  {len(all_images)} images in [bold]{elapsed:.1f}s[/bold]  "
        f"([dim]{rate:.1f} img/s[/dim])\n"
    )

    # ── Summary table ────────────────────────────────────────────────────────
    console.print(Rule("Results", style="cyan"))
    print_summary(tagged)
    console.print()

    # ── Write tags.json ───────────────────────────────────────────────────────
    tags_out = Path(args.tags_path)
    payload  = {
        "meta": {
            "version":       "1.0",
            "generated_at":  datetime.now().isoformat(timespec="seconds"),
            "source_folder": str(src),
            "recursive":     args.recursive,
            "clip_model":    args.clip_model,
            "threshold":     args.threshold,
            "total_images":  len(all_images),
            "scene_labels":  SCENE_LABELS,
            "theme_labels":  THEME_LABELS,
        },
        "images": tagged,
    }
    tags_out.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    console.print(f"  [green]✓[/green]  Tags  →  [link={tags_out.as_uri()}]{tags_out}[/link]\n")

    # ── Group ─────────────────────────────────────────────────────────────────
    if args.group:
        console.print(Rule("Grouping", style="cyan"))
        if args.execute:
            verb = "Copying" if getattr(args, "copy", False) else "Moving"
            with console.status(f"[cyan]{verb} files …[/cyan]"):
                moved, errors = action_group(tagged, src, copy_mode=getattr(args, "copy", False))
            console.print(
                f"  [green]✓[/green]  {moved} files {'copied' if getattr(args,'copy',False) else 'moved'}"
                f"  ·  {errors} errors\n"
            )
        else:
            # Dry-run preview: show what would happen
            preview: dict[str, int] = defaultdict(int)
            for rec in tagged:
                if "error" not in rec and rec["scene"]["accepted"] and rec["theme"]["accepted"]:
                    preview[f"{_slugify(rec['scene']['label'])}/{_slugify(rec['theme']['label'])}"] += 1
                elif "error" not in rec:
                    preview["_unclassified"] += 1
            console.print("  [yellow]DRY RUN[/yellow] — would create folders:\n")
            for folder, count in sorted(preview.items(), key=lambda x: -x[1]):
                console.print(f"    [dim]{src / folder}[/dim]  →  {count} images")
            console.print()

    # ── Rename ────────────────────────────────────────────────────────────────
    if args.rename:
        console.print(Rule("Renaming", style="cyan"))
        if args.execute:
            with console.status("[cyan]Renaming files …[/cyan]"):
                renamed, errors = action_rename(tagged)
            console.print(f"  [green]✓[/green]  {renamed} files renamed  ·  {errors} errors\n")
            # Re-save tags.json with updated paths/names
            tags_out.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        else:
            sample = [
                f"  [dim]{Path(r['path']).name}[/dim]  →  "
                f"[white]{r['suggested_name']}_0001{Path(r['path']).suffix}[/white]"
                for r in tagged[:5] if "error" not in r
            ]
            console.print("  [yellow]DRY RUN[/yellow] — rename preview (first 5):\n")
            for line in sample:
                console.print(line)
            console.print()

    # ── Done ──────────────────────────────────────────────────────────────────
    n_ok      = sum(1 for r in tagged if "error" not in r)
    n_grouped = len({r.get("group_key") for r in tagged if "error" not in r})

    console.print(Panel.fit(
        Text.assemble(
            ("Done!\n", "bold green"),
            (f"  {n_ok} images tagged  ·  {n_grouped} distinct groups\n", "white"),
            (f"  tags.json  →  {tags_out}", "dim"),
        ),
        border_style="green",
    ))
    console.print()


if __name__ == "__main__":
    main()
