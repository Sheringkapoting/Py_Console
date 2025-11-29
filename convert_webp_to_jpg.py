#!/usr/bin/env python3
import os
import sys
import struct
from typing import List, Tuple, Optional
from PIL import Image, UnidentifiedImageError, ImageFile, ImageOps
from tqdm import tqdm

# Safety: mitigate decompression bombs
ImageFile.LOAD_TRUNCATED_IMAGES = False
Image.MAX_IMAGE_PIXELS = 50_000_000  # ~50MP cap; adjust if needed

CONVERTIBLE_EXTS = ('.webp', '.png', '.jpeg')
# Explicitly exclude media/video and GIF (animated image) by extension and magic
EXCLUDE_EXTS = ('.mp4', '.mov', '.gif', '.avi', '.mkv')

def list_files_recursive(folder: str) -> List[str]:
    files: List[str] = []
    for root, _, fnames in os.walk(folder, topdown=True):
        for fname in fnames:
            files.append(os.path.join(root, fname))
    return files

def _read_head(path: str, size: int = 64) -> bytes:
    try:
        with open(path, 'rb') as f:
            return f.read(size)
    except Exception:
        return b''

def detect_magic_type(path: str) -> str:
    """Return a coarse type based on magic numbers: 'webp','png','jpeg','gif','mp4','avi','mkv','unknown'."""
    head = _read_head(path, 64)
    if not head or len(head) < 12:
        return 'unknown'
    # PNG
    if head.startswith(b"\x89PNG\r\n\x1a\n"):
        return 'png'
    # JPEG
    if head.startswith(b"\xFF\xD8\xFF"):
        return 'jpeg'
    # GIF
    if head.startswith(b"GIF87a") or head.startswith(b"GIF89a"):
        return 'gif'
    # RIFF containers: WEBP and AVI
    if head.startswith(b"RIFF") and len(head) >= 12:
        fourcc = head[8:12]
        if fourcc == b"WEBP":
            return 'webp'
        if fourcc == b"AVI ":
            return 'avi'
    # MP4/MOV (ISO Base Media): look for 'ftyp' box at offset 4
    if len(head) >= 12 and head[4:8] == b"ftyp":
        return 'mp4'
    # Matroska / WebM (EBML)
    if head.startswith(b"\x1A\x45\xDF\xA3"):
        return 'mkv'
    return 'unknown'

def is_supported_image(path: str) -> bool:
    """Strict filter: require allowed extension and matching magic type; explicitly exclude videos/GIF."""
    lower = path.lower()
    if lower.endswith(EXCLUDE_EXTS):
        return False
    # Extension must be strictly one of convertible set
    if not lower.endswith(CONVERTIBLE_EXTS):
        return False
    t = detect_magic_type(path)
    return t in ('webp', 'png', 'jpeg')

def target_jpg_path(src_path: str) -> str:
    """Return the default JPG path (no conflict handling).

    This preserves the original base name and directory, changing only the
    extension to `.jpg`. Backward-compatible helper retained for internal use.
    """
    root, _ = os.path.splitext(src_path)
    return root + ".jpg"

def generate_conflict_free_jpg_path(src_path: str) -> str:
    """Generate a conflict-free `.jpg` path in the same directory.

    Behavior:
    - If `<basename>.jpg` does not exist, use it as-is.
    - If it exists, append an incremental number before the extension:
      `<basename>(1).jpg`, `<basename>(2).jpg`, ... until an available name
      is found.

    This maintains the original base name and directory, and applies uniformly
    to WEBP/PNG/JPEG sources. Only the output filename changes when necessary.
    """
    # Derive directory and base name
    dir_name = os.path.dirname(src_path)
    base_name = os.path.splitext(os.path.basename(src_path))[0]

    # First try the natural target
    candidate = os.path.join(dir_name, f"{base_name}.jpg")
    if not os.path.exists(candidate):
        return candidate

    # Incrementally try numbered variants
    n = 1
    while True:
        candidate = os.path.join(dir_name, f"{base_name}({n}).jpg")
        if not os.path.exists(candidate):
            return candidate
        n += 1

def _webp_has_animation_vp8x(head: bytes) -> Optional[bool]:
    # Minimal parsing to inspect VP8X feature bits and ANIM chunk presence
    try:
        if not head.startswith(b"RIFF") or head[8:12] != b"WEBP":
            return None
        # We need to parse chunks; read more for safety
        return None  # We will parse in full file function below
    except Exception:
        return None

def is_webp_animated(path: str) -> bool:
    """Detect WebP animation via VP8X feature bits or presence of ANIM chunk."""
    try:
        with open(path, 'rb') as f:
            data = f.read(1024 * 64)  # read first 64KB which should include headers
        if not data.startswith(b"RIFF") or data[8:12] != b"WEBP":
            return False
        # RIFF layout: 'RIFF' size(4) 'WEBP' chunks...
        offset = 12
        data_len = len(data)
        while offset + 8 <= data_len:
            chunk_id = data[offset:offset+4]
            chunk_size = struct.unpack('<I', data[offset+4:offset+8])[0]
            payload_start = offset + 8
            payload_end = payload_start + chunk_size
            # Check for VP8X features
            if chunk_id == b'VP8X' and payload_start + 1 <= data_len:
                features = data[payload_start]
                # Bit 1 indicates animation
                if (features & 0x02) != 0:
                    return True
            # Check for ANIM chunk directly
            if chunk_id == b'ANIM':
                return True
            # Chunks are padded to even sizes
            padded = chunk_size + (chunk_size % 2)
            offset = payload_start + padded
        return False
    except Exception:
        # If we cannot parse, err on the side of not animated to avoid false skips
        return False

def convert_one(file_path: str) -> Tuple[bool, str]:
    """
    Returns (ok, err_msg). On success, ok=True and err_msg="".
    On failure or skip, ok=False and err_msg contains brief text.
    """
    try:
        # Validate quickly before loading fully
        if not os.path.isfile(file_path):
            return (False, "not a file")
        size = os.path.getsize(file_path)
        if size <= 0:
            return (False, "zero size")

        # Strict filter: extension + magic
        if not is_supported_image(file_path):
            return (False, "unsupported type or extension")

        # WebP: skip animated
        if detect_magic_type(file_path) == 'webp' and is_webp_animated(file_path):
            return (False, "animated webp: skipped")

        # Convert to high-quality RGB JPG, preserve orientation and metadata
        with Image.open(file_path) as im:
            # Verify content
            try:
                im.verify()
            except (UnidentifiedImageError, Image.DecompressionBombError) as e:
                return (False, f"invalid image: {e}")
        # Re-open to actually convert after verify() (Pillow requirement)
        with Image.open(file_path) as im2:
            # Preserve orientation
            im2 = ImageOps.exif_transpose(im2)
            rgb = im2.convert("RGB")
            # Determine a conflict-free output path in the same directory
            out_path = generate_conflict_free_jpg_path(file_path)
            if os.path.abspath(out_path) == os.path.abspath(file_path):
                return (False, "same path as source")

            save_kwargs = {
                'quality': 95,
                'subsampling': 0,  # 4:4:4 for better quality
                'optimize': True,
                'progressive': True,
            }
            exif = im2.info.get('exif')
            if exif:
                save_kwargs['exif'] = exif
            icc = im2.info.get('icc_profile')
            if icc:
                save_kwargs['icc_profile'] = icc

            rgb.save(out_path, "JPEG", **save_kwargs)
        return (True, "")
    except Exception as e:
        return (False, str(e))

def convert_images(folder: str) -> None:
    if not os.path.isdir(folder):
        print(f"[ERROR] The path '{folder}' is not a valid directory.")
        return

    all_files = list_files_recursive(folder)
    total_candidates = [f for f in all_files if f.lower().endswith(CONVERTIBLE_EXTS)]
    total = len(total_candidates)

    if total == 0:
        print("[INFO] No convertible images found (.webp, .png, .jpeg).")
        return

    converted = 0
    skipped = 0
    failed = 0

    # Progress bar with running postfix: Converted=X | Deleted=Y
    with tqdm(total=total, desc="Converting", unit="img") as pbar:
        for file_path in total_candidates:
            # Large-file note in postfix (not per-byte progress)
            size_mb = 0
            try:
                size_mb = os.path.getsize(file_path) / (1024 * 1024)
            except Exception:
                size_mb = 0

            ok, err = convert_one(file_path)
            if ok:
                converted += 1
            else:
                # Treat animated WebP and unsupported types as skip, others as failure
                if err.startswith("animated webp") or err.startswith("unsupported type"):
                    skipped += 1
                else:
                    failed += 1

            pbar.set_postfix_str(
                f"Converted={converted} | Skipped={skipped} | Failed={failed} | Size={size_mb:.1f}MB"
            )
            pbar.update(1)

    # Final summary
    print("\n[SUMMARY]")
    print(f" Total files considered: {total}")
    print(f" Converted: {converted}")
    print(f" Skipped: {skipped}")
    print(f" Failed: {failed}")

if __name__ == "__main__":
    # Accept folder path from command line or prompt
    if len(sys.argv) > 1:
        folder_path = sys.argv[1]
    else:
        folder_path = input("Enter folder path to convert images: ").strip()
    convert_images(folder_path)
