#!/usr/bin/env python3
import os
import sys
from typing import List, Tuple
from PIL import Image, UnidentifiedImageError, ImageFile
from tqdm import tqdm
try:
    import msvcrt
    _HAS_MSVC = True
except Exception:
    _HAS_MSVC = False

# Safety: mitigate decompression bombs
ImageFile.LOAD_TRUNCATED_IMAGES = False
Image.MAX_IMAGE_PIXELS = 80_000_000  # ~50MP cap; adjust if needed

CONVERTIBLE_EXTS = ('.webp', '.png', '.jpeg')
SKIP_EXTS = ('.gif', '.jpg')  # .jpg is already target; .gif we skip per original code

def is_webp_animated(path: str) -> Tuple[bool, str]:
    try:
        with open(path, 'rb') as f:
            header = f.read(12)
            if len(header) < 12:
                return (False, "truncated header")
            if header[0:4] != b'RIFF' or header[8:12] != b'WEBP':
                return (False, "not webp riff")
            while True:
                chunk_hdr = f.read(8)
                if len(chunk_hdr) < 8:
                    break
                fourcc = chunk_hdr[0:4]
                size = int.from_bytes(chunk_hdr[4:8], 'little')
                if fourcc == b'VP8X':
                    data = f.read(size)
                    if len(data) < size:
                        return (False, "truncated vp8x")
                    flags = data[0]
                    if flags & 0x02:
                        return (True, "vp8x animated")
                elif fourcc == b'ANIM':
                    f.seek(size, os.SEEK_CUR)
                    return (True, "anim chunk")
                else:
                    f.seek(size, os.SEEK_CUR)
                if size % 2 == 1:
                    f.seek(1, os.SEEK_CUR)
        return (False, "")
    except Exception as e:
        return (False, str(e))

def list_files_recursive(folder: str) -> List[str]:
    files: List[str] = []
    for root, _, fnames in os.walk(folder, topdown=True):
        for fname in fnames:
            files.append(os.path.join(root, fname))
    return files

def should_convert(path: str) -> bool:
    lower = path.lower()
    if lower.endswith(SKIP_EXTS):
        return False
    if lower.endswith('.webp'):
        animated, _ = is_webp_animated(path)
        if animated:
            return False
    return lower.endswith(CONVERTIBLE_EXTS)

def target_jpg_path(src_path: str) -> str:
    root, _ = os.path.splitext(src_path)
    return root + ".jpg"

def convert_one(file_path: str) -> Tuple[bool, str]:
    """
    Returns (ok, err_msg). On success, ok=True and err_msg="".
    On failure, ok=False and err_msg contains brief text.
    """
    try:
        # Validate quickly before loading fully
        if not os.path.isfile(file_path):
            return (False, "not a file")
        size = os.path.getsize(file_path)
        if size <= 0:
            return (False, "zero size")
        if file_path.lower().endswith('.webp'):
            animated, _ = is_webp_animated(file_path)
            if animated:
                return (False, "animated webp")

        # Convert to RGB JPG
        with Image.open(file_path) as im:
            # Verify content
            try:
                im.verify()
            except (UnidentifiedImageError, Image.DecompressionBombError) as e:
                return (False, f"invalid image: {e}")
        # Re-open to actually convert after verify() (Pillow requirement)
        with Image.open(file_path) as im2:
            if im2.mode in ("RGBA", "LA") or (im2.mode == "P" and "transparency" in im2.info):
                rgba = im2.convert("RGBA")
                bg = Image.new("RGB", rgba.size, (255, 255, 255))
                bg.paste(rgba, mask=rgba.split()[3])
                rgb = bg
            else:
                rgb = im2.convert("RGB")
            out_path = target_jpg_path(file_path)
            # Ensure we do not overwrite an existing different file unintentionally
            if os.path.abspath(out_path) == os.path.abspath(file_path):
                # Shouldn’t happen because we skip .jpg inputs
                return (False, "same path as source")
            rgb.save(out_path, "JPEG")
        return (True, "")
    except Exception as e:
        return (False, str(e))

def delete_source(file_path: str) -> Tuple[bool, str]:
    try:
        os.remove(file_path)
        return (True, "")
    except Exception as e:
        return (False, str(e))

def convert_images(folder: str) -> None:
    if not os.path.isdir(folder):
        print(f"[ERROR] The path '{folder}' is not a valid directory.")
        return

    all_files = list_files_recursive(folder)
    total_candidates = [f for f in all_files if should_convert(f)]
    total = len(total_candidates)

    if total == 0:
        print("[INFO] No convertible images found (.webp, .png, .jpeg).")
        return

    converted = 0
    deleted = 0
    failed = 0

    # Progress bar with running postfix: Converted=X | Deleted=Y
    with tqdm(total=total, desc="Converting", unit="img", dynamic_ncols=True, mininterval=0.3, file=sys.stdout) as pbar:
        for file_path in total_candidates:
            ok, err = convert_one(file_path)
            if ok:
                # Only delete original if conversion succeeded
                del_ok, _ = delete_source(file_path)
                converted += 1
                if del_ok:
                    deleted += 1
                else:
                    # Deletion failure is not counted as conversion failure; we still converted successfully
                    pass
            else:
                failed += 1

            pbar.set_postfix_str(f"Converted={converted} | Deleted={deleted}")
            pbar.update(1)

    # Final summary
    print("\n[SUMMARY]")
    print(f" Total files considered: {total}")
    print(f" Converted: {converted}")
    print(f" Deleted originals: {deleted}")
    print(f" Failed: {failed}")

def _jpg_paths(folder: str) -> List[str]:
    return [f for f in list_files_recursive(folder) if f.lower().endswith('.jpg')]

def compress_one_jpg(file_path: str) -> Tuple[bool, str, bool]:
    try:
        if not os.path.isfile(file_path):
            return (False, "not a file", False)
        orig_size = os.path.getsize(file_path)
        if orig_size <= 0:
            return (False, "zero size", False)
        tmp_path = file_path + ".tmp"
        with Image.open(file_path) as im:
            exif = im.info.get("exif")
            icc = im.info.get("icc_profile")
            rgb = im.convert("RGB")
            rgb.save(
                tmp_path,
                "JPEG",
                quality=85,
                optimize=True,
                progressive=True,
                subsampling=1,
                exif=exif,
                icc_profile=icc,
            )
        new_size = os.path.getsize(tmp_path)
        if new_size < orig_size:
            os.replace(tmp_path, file_path)
            return (True, "", True)
        else:
            try:
                os.remove(tmp_path)
            except Exception:
                pass
            return (True, "no reduction", False)
    except KeyboardInterrupt:
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass
        raise
    except Exception as e:
        try:
            if 'tmp_path' in locals() and os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass
        return (False, str(e), False)

def compress_jpgs(folder: str) -> None:
    jpgs = _jpg_paths(folder)
    total = len(jpgs)
    if total == 0:
        print("[INFO] No .jpg files found to compress.")
        return
    compressed = 0
    deleted = 0
    skipped = 0
    failed = 0
    aborted = False
    try:
        with tqdm(total=total, desc="Compressing", unit="img", dynamic_ncols=True, mininterval=0.3, file=sys.stdout) as pbar:
            for file_path in jpgs:
                if _HAS_MSVC and msvcrt.kbhit():
                    key = msvcrt.getch()
                    if key in (b"\x1b",):
                        aborted = True
                        break
                ok, err, replaced = compress_one_jpg(file_path)
                if ok:
                    if replaced:
                        compressed += 1
                        deleted += 1
                    else:
                        skipped += 1
                else:
                    failed += 1
                pbar.set_postfix_str(
                    f"Compressed={compressed} | Deleted={deleted} | Skipped={skipped}"
                )
                pbar.update(1)
    except KeyboardInterrupt:
        aborted = True
    if aborted:
        print("\n[INFO] Compression cancelled by user.")
    print("\n[SUMMARY]")
    print(f" Total .jpg files considered: {total}")
    print(f" Compressed: {compressed}")
    print(f" Deleted originals: {deleted}")
    print(f" Skipped: {skipped}")
    print(f" Failed: {failed}")

if __name__ == "__main__":
    # Accept folder path from command line or prompt
    if len(sys.argv) > 1:
        folder_path = sys.argv[1]
    else:
        folder_path = input("Enter folder path to convert images: ").strip()
    convert_images(folder_path)
    try:
        resp = input("Would you like to run the compression process? (yes/no) ").strip().lower()
    except KeyboardInterrupt:
        print("\n[INFO] Compression process prompt cancelled by user.")
        sys.exit(0)
    if resp in ("y", "yes"):
        compress_jpgs(folder_path)
    else:
        print("[INFO] Compression skipped.")
def is_webp_animated(path: str) -> Tuple[bool, str]:
    try:
        with open(path, 'rb') as f:
            header = f.read(12)
            if len(header) < 12:
                return (False, "truncated header")
            if header[0:4] != b'RIFF' or header[8:12] != b'WEBP':
                return (False, "not webp riff")
            while True:
                chunk_hdr = f.read(8)
                if len(chunk_hdr) < 8:
                    break
                fourcc = chunk_hdr[0:4]
                size = int.from_bytes(chunk_hdr[4:8], 'little')
                if fourcc == b'VP8X':
                    data = f.read(size)
                    if len(data) < size:
                        return (False, "truncated vp8x")
                    flags = data[0]
                    if flags & 0x02:
                        return (True, "vp8x animated")
                elif fourcc == b'ANIM':
                    f.seek(size, os.SEEK_CUR)
                    return (True, "anim chunk")
                else:
                    f.seek(size, os.SEEK_CUR)
                if size % 2 == 1:
                    f.seek(1, os.SEEK_CUR)
        return (False, "")
    except Exception as e:
        return (False, str(e))
