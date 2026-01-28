#!/usr/bin/env python3
import os
import sys
import time
from typing import List, Tuple
from PIL import Image, UnidentifiedImageError, ImageFile
from tqdm import tqdm
try:
    import msvcrt
    _HAS_MSVC = True
except Exception:
    _HAS_MSVC = False

# HEIC support
try:
    from pillow_heif import register_heif_opener
    register_heif_opener()
    _HAS_HEIC = True
except ImportError:
    _HAS_HEIC = False
    print("[WARN] pillow-heif not found. .heic support disabled.")

# Safety: mitigate decompression bombs
ImageFile.LOAD_TRUNCATED_IMAGES = False
Image.MAX_IMAGE_PIXELS = 80_000_000  # ~50MP cap; adjust if needed

CONVERTIBLE_EXTS = ['.webp', '.png', '.jpeg']
if _HAS_HEIC:
    CONVERTIBLE_EXTS.append('.heic')
CONVERTIBLE_EXTS = tuple(CONVERTIBLE_EXTS)
SKIP_EXTS = ('.gif', '.jpg')  # .jpg is already target; .gif we skip per original code

def is_heic_animated(path: str) -> Tuple[bool, str]:
    """Check if HEIC file is animated or has multiple frames."""
    if not _HAS_HEIC:
        return (False, "heic support not available")
    
    try:
        with Image.open(path) as im:
            # Check if image is animated or has multiple frames
            is_animated = getattr(im, 'is_animated', False)
            n_frames = getattr(im, 'n_frames', 1)
            
            if is_animated or n_frames > 1:
                return (True, f"animated heic ({n_frames} frames)")
            return (False, "")
    except Exception as e:
        return (False, str(e))

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
    if lower.endswith('.heic'):
        animated, _ = is_heic_animated(path)
        if animated:
            return False
    return lower.endswith(CONVERTIBLE_EXTS)

def target_jpg_path(src_path: str) -> str:
    root, _ = os.path.splitext(src_path)
    return root + ".jpg"

def format_size(size_bytes: int) -> str:
    """Format bytes in human readable format."""
    if size_bytes == 0:
        return "0B"
    size_names = ["B", "KB", "MB", "GB"]
    i = 0
    while size_bytes >= 1024.0 and i < len(size_names) - 1:
        size_bytes /= 1024.0
        i += 1
    return f"{size_bytes:.1f}{size_names[i]}"

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
        if file_path.lower().endswith('.heic'):
            animated, _ = is_heic_animated(file_path)
            if animated:
                return (False, "animated heic")

        # Convert to RGB JPG
        with Image.open(file_path) as im:
            # Check for HEIC animation/multi-frame
            if _HAS_HEIC and file_path.lower().endswith('.heic'):
                if getattr(im, 'is_animated', False) or getattr(im, 'n_frames', 1) > 1:
                    return (False, "animated/multi-frame heic")

            # Verify content
            try:
                im.verify()
            except (UnidentifiedImageError, Image.DecompressionBombError) as e:
                return (False, f"invalid image: {e}")
        # Re-open to actually convert after verify() (Pillow requirement)
        with Image.open(file_path) as im2:
            exif = im2.info.get("exif")
            icc = im2.info.get("icc_profile")

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

            save_kwargs = {
                "quality": 85,
                "optimize": True,
                "progressive": True,
                "subsampling": 2,
            }
            if exif:
                save_kwargs["exif"] = exif
            if icc:
                save_kwargs["icc_profile"] = icc

            rgb.save(out_path, "JPEG", **save_kwargs)
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
        print("[INFO] No convertible images found (.webp, .png, .jpeg" + (", .heic" if _HAS_HEIC else "") + ").")
        return

    converted = 0
    deleted = 0
    failed = 0
    total_original_size = 0
    total_converted_size = 0
    start_time = time.time()

    # Enhanced progress bar with detailed metrics
    with tqdm(total=total, desc="Converting", unit="img", dynamic_ncols=True, mininterval=0.2, file=sys.stdout,
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] {rate_fmt}") as pbar:
        for file_path in total_candidates:
            file_start_time = time.time()
            original_size = os.path.getsize(file_path) if os.path.exists(file_path) else 0
            total_original_size += original_size
            
            ok, err = convert_one(file_path)
            
            if ok:
                converted_size = os.path.getsize(target_jpg_path(file_path)) if os.path.exists(target_jpg_path(file_path)) else 0
                total_converted_size += converted_size
                
                # Only delete original if conversion succeeded
                del_ok, _ = delete_source(file_path)
                converted += 1
                if del_ok:
                    deleted += 1
                
                # Calculate compression ratio for this file
                compression_ratio = (1 - converted_size / original_size) * 100 if original_size > 0 else 0
                ratio_str = f"{compression_ratio:.1f}%"
            else:
                failed += 1
                ratio_str = "N/A"
                converted_size = 0

            # Calculate processing rate and time estimates
            elapsed = time.time() - start_time
            processed = converted + failed
            rate = processed / elapsed if elapsed > 0 else 0
            
            # Enhanced postfix with comprehensive metrics
            postfix_items = [
                f"✓{converted}",
                f"✗{failed}",
                f"🗑{deleted}",
                f"📁{format_size(total_original_size)}",
                f"📦{format_size(total_converted_size)}"
            ]
            
            if converted > 0 and total_original_size > 0:
                overall_ratio = (1 - total_converted_size / total_original_size) * 100
                postfix_items.append(f"💾{overall_ratio:.1f}%")
            
            postfix_items.append(f"⚡{rate:.1f}/s")
            
            pbar.set_postfix_str(" | ".join(postfix_items))
            pbar.update(1)

    # Enhanced final summary
    elapsed_time = time.time() - start_time
    print("\n" + "="*60)
    print("📊 CONVERSION SUMMARY")
    print("="*60)
    print(f"📁 Total files processed:     {total}")
    print(f"✅ Successfully converted:     {converted}")
    print(f"🗑️  Original files deleted:    {deleted}")
    print(f"❌ Failed conversions:        {failed}")
    print(f"⏱️  Total processing time:     {elapsed_time:.1f}s")
    
    if total_original_size > 0:
        print(f"💾 Original total size:        {format_size(total_original_size)}")
        print(f"📦 Converted total size:       {format_size(total_converted_size)}")
        space_saved = total_original_size - total_converted_size
        print(f"💰 Space saved:               {format_size(space_saved)} ({(1 - total_converted_size/total_original_size)*100:.1f}%)")
        
        if converted > 0:
            avg_rate = converted / elapsed_time if elapsed_time > 0 else 0
            print(f"⚡ Average processing rate:   {avg_rate:.1f} files/sec")
    
    print("="*60)

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

            save_kwargs = {
                "quality": 85,
                "optimize": True,
                "progressive": True,
                "subsampling": 2,
            }
            if exif:
                save_kwargs["exif"] = exif
            if icc:
                save_kwargs["icc_profile"] = icc

            rgb.save(tmp_path, "JPEG", **save_kwargs)
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
    
    total_original_size = sum(os.path.getsize(f) for f in jpgs if os.path.exists(f))
    total_compressed_size = 0
    total_saved = 0
    start_time = time.time()
    
    try:
        with tqdm(total=total, desc="Compressing", unit="img", dynamic_ncols=True, mininterval=0.2, file=sys.stdout,
                    bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] {rate_fmt}") as pbar:
            for file_path in jpgs:
                if _HAS_MSVC and msvcrt.kbhit():
                    key = msvcrt.getch()
                    if key in (b"\x1b",):
                        aborted = True
                        break
                
                original_size = os.path.getsize(file_path) if os.path.exists(file_path) else 0
                ok, err, replaced = compress_one_jpg(file_path)
                
                if ok:
                    if replaced:
                        compressed += 1
                        deleted += 1
                        new_size = os.path.getsize(file_path) if os.path.exists(file_path) else 0
                        saved = original_size - new_size
                        total_saved += saved
                        total_compressed_size += new_size
                    else:
                        skipped += 1
                        total_compressed_size += original_size
                else:
                    failed += 1
                    total_compressed_size += original_size
                
                # Calculate processing metrics
                elapsed = time.time() - start_time
                processed = compressed + skipped + failed
                rate = processed / elapsed if elapsed > 0 else 0
                
                # Enhanced postfix with compression metrics
                postfix_items = [
                    f"🗜{compressed}",
                    f"⏭{skipped}",
                    f"✗{failed}",
                    f"💾{format_size(total_saved)}"
                ]
                
                if total_saved > 0 and total_original_size > 0:
                    avg_compression = (total_saved / total_original_size) * 100
                    postfix_items.append(f"📉{avg_compression:.1f}%")
                
                postfix_items.append(f"⚡{rate:.1f}/s")
                
                pbar.set_postfix_str(" | ".join(postfix_items))
                pbar.update(1)
    except KeyboardInterrupt:
        aborted = True
    
    # Enhanced compression summary
    elapsed_time = time.time() - start_time
    if aborted:
        print("\n[INFO] Compression cancelled by user.")
    
    print("\n" + "="*60)
    print("📊 COMPRESSION SUMMARY")
    print("="*60)
    print(f"📁 Total .jpg files processed:  {total}")
    print(f"🗜️  Successfully compressed:      {compressed}")
    print(f"⏭️  Skipped (no reduction):      {skipped}")
    print(f"❌ Failed compressions:         {failed}")
    print(f"⏱️  Total processing time:        {elapsed_time:.1f}s")
    
    if total_original_size > 0:
        print(f"💾 Original total size:          {format_size(total_original_size)}")
        print(f"📦 Final total size:             {format_size(total_compressed_size)}")
        print(f"💰 Total space saved:            {format_size(total_saved)} ({(total_saved/total_original_size)*100:.1f}%)")
        
        if compressed > 0:
            avg_rate = compressed / elapsed_time if elapsed_time > 0 else 0
            print(f"⚡ Average compression rate:    {avg_rate:.1f} files/sec")
    
    print("="*60)

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

