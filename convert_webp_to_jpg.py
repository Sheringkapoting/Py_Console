#!/usr/bin/env python3
import os
import sys
from typing import List, Tuple
from PIL import Image, UnidentifiedImageError, ImageFile
from tqdm import tqdm

# Safety: mitigate decompression bombs
ImageFile.LOAD_TRUNCATED_IMAGES = False
Image.MAX_IMAGE_PIXELS = 50_000_000  # ~50MP cap; adjust if needed

CONVERTIBLE_EXTS = ('.webp', '.png', '.jpeg')
SKIP_EXTS = ('.gif', '.jpg')  # .jpg is already target; .gif we skip per original code

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

        # Convert to RGB JPG
        with Image.open(file_path) as im:
            # Verify content
            try:
                im.verify()
            except (UnidentifiedImageError, Image.DecompressionBombError) as e:
                return (False, f"invalid image: {e}")
        # Re-open to actually convert after verify() (Pillow requirement)
        with Image.open(file_path) as im2:
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
    with tqdm(total=total, desc="Converting", unit="img") as pbar:
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

if __name__ == "__main__":
    # Accept folder path from command line or prompt
    if len(sys.argv) > 1:
        folder_path = sys.argv[1]
    else:
        folder_path = input("Enter folder path to convert images: ").strip()
    convert_images(folder_path)
