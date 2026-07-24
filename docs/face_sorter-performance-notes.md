# face_sorter.py — CPU Optimization Notes

> Relocated from the old root-level `CPU_Optimization.md` during the Phase 0
> documentation cleanup. Line numbers below were captured against an earlier
> revision of `face_sorter.py` and have not been reverified against the
> current file — treat them as approximate pointers, not exact line refs,
> and re-locate the relevant code before acting on any item here.

After analyzing `face_sorter.py`, here are the key CPU bottlenecks and optimization suggestions:

## Primary CPU Bottlenecks

### 1. Face Detection Processing
- **Issue**: `app.get(img)` processes every image at full 640x640 resolution
- **Impact**: Most CPU-intensive operation, runs on every image
- **Suggestion**:
  - Add image size preprocessing to resize large images before detection
  - Implement early image filtering (skip very small/large images)
  - Consider reducing detection resolution from 640x640 to 320x320 for faster processing

### 2. Embedding Comparison Loop
- **Issue**: Nested loops comparing each face to all reference embeddings
- **Impact**: O(n×m) complexity where n=faces per image, m=reference embeddings
- **Suggestion**:
  - Pre-compute and cache reference embedding statistics
  - Use vectorized operations with NumPy instead of Python loops
  - Implement early termination when similarity exceeds threshold

### 3. Synchronous Processing
- **Issue**: Processes images sequentially in a single thread
- **Impact**: Underutilizes multi-core CPUs
- **Suggestion**:
  - Implement batch processing with ThreadPoolExecutor
  - Use multiprocessing for CPU-bound face detection
  - Add configurable worker count based on CPU cores

### 4. Redundant Image Loading
- **Issue**: Images loaded multiple times for different operations
- **Impact**: Unnecessary I/O and memory overhead
- **Suggestion**:
  - Cache loaded images in memory with LRU eviction
  - Pre-load and batch process images in chunks

## Secondary Optimizations

### 5. Progress Bar Overhead
- **Issue**: High refresh rate (6Hz) creates unnecessary updates
- **Suggestion**: Reduce refresh rate to 1-2Hz for large batches

### 6. HTML Report Generation
- **Issue**: Base64 encoding and thumbnail generation for every image
- **Suggestion**:
  - Move report generation to a separate thread
  - Generate thumbnails only for matched images
  - Use more efficient image compression

### 7. Memory Management
- **Issue**: No memory limits for large image collections
- **Suggestion**:
  - Implement batch processing with configurable batch sizes
  - Add memory usage monitoring and throttling

## Quick Wins (Highest Impact)
1. **Reduce detection resolution** from 640x640 to 320x320 (50-70% speedup)
2. **Add multiprocessing** for face detection (2-4x speedup on multi-core)
3. **Implement early image filtering** to skip obvious non-face images
4. **Cache reference embeddings** to avoid redundant computations

## Configuration Additions to Consider
- `--batch-size` for memory management
- `--workers` for parallel processing
- `--detection-size` for resolution control
- `--max-image-size` to skip overly large images

These optimizations could reduce CPU usage from 80%+ to 20-40% while maintaining accuracy, especially for large image collections.
