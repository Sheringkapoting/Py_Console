# Image Processor Suite - Usage Guide

## Quick Start

### Installation

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Verify Installation**:
   ```bash
   python test_basic.py
   ```

### Running the Application

#### GUI Mode (Recommended)
```bash
python src/image_processor_suite/main.py
```

#### Command Line Mode

**Image Conversion**:
```bash
# Convert all images in a folder
python src/image_processor_suite/main.py convert /path/to/images --recursive

# Convert with custom quality and output directory
python src/image_processor_suite/main.py convert /path/to/images --output /path/to/output --quality 90

# Keep original files
python src/image_processor_suite/main.py convert /path/to/images --keep-originals
```

**Face Recognition Sorting**:
```bash
# Sort images by one reference face
python src/image_processor_suite/main.py sort /path/to/images --face /path/to/reference.jpg --dest /path/to/sorted

# Sort by multiple faces
python src/image_processor_suite/main.py sort /path/to/images \
  --face /path/to/person1.jpg --dest /path/to/person1_folder \
  --face /path/to/person2.jpg --dest /path/to/person2_folder

# Adjust tolerance for matching
python src/image_processor_suite/main.py sort /path/to/images \
  --face /path/to/reference.jpg --dest /path/to/sorted --tolerance 0.4
```

## Features Overview

### 🖼️ Image Conversion
- **Supported Input Formats**: WEBP, PNG, JPEG, BMP, TIFF
- **Output Format**: High-quality JPG
- **Batch Processing**: Process entire directories recursively
- **Quality Control**: Adjustable JPEG quality (1-100)
- **Safety Features**: File validation, size limits, error recovery
- **Performance**: Multi-threaded processing

### 👤 Face Recognition & Sorting
- **Advanced Face Detection**: Uses state-of-the-art face recognition models
- **Multiple Reference Faces**: Sort by multiple people simultaneously
- **Adjustable Tolerance**: Fine-tune matching sensitivity
- **Secure Operations**: Path traversal protection, safe file moves
- **Parallel Processing**: Multi-process face analysis for speed
- **Comprehensive Logging**: Detailed operation logs

### 🖥️ User Interface
- **Professional GUI**: Modern tkinter-based interface
- **Progress Tracking**: Real-time progress bars and status updates
- **Settings Management**: Persistent configuration
- **Error Handling**: User-friendly error messages
- **Integrated Workflow**: Seamless switching between functions

## Configuration

### Settings File
The application stores settings in `config/settings.json` (auto-created):

```json
{
  "conversion": {
    "quality": 95,
    "delete_originals": true,
    "recursive": true,
    "max_workers": 4
  },
  "face_recognition": {
    "tolerance": 0.5,
    "jitter": 1,
    "max_workers": null,
    "algorithm": "euclidean"
  },
  "gui": {
    "theme": "default",
    "window_width": 800,
    "window_height": 600
  }
}
```

### Environment Variables
- `IMAGE_PROCESSOR_LOG_LEVEL`: Set logging level (DEBUG, INFO, WARNING, ERROR)
- `IMAGE_PROCESSOR_MAX_WORKERS`: Override default worker count

## Troubleshooting

### Common Issues

1. **"No module named 'face_recognition'"**
   - Install face_recognition: `pip install face_recognition`
   - On Windows, you may need Visual Studio Build Tools

2. **"GUI not available"**
   - tkinter is not installed or available
   - Use CLI mode instead: `python src/image_processor_suite/main.py convert --help`

3. **"Permission denied" errors**
   - Check file/folder permissions
   - Run as administrator if necessary
   - Ensure output directories are writable

4. **Face recognition not finding faces**
   - Try adjusting tolerance (lower = stricter, higher = more lenient)
   - Ensure reference images have clear, front-facing faces
   - Check image quality and lighting

5. **Slow performance**
   - Reduce number of workers if system is overloaded
   - Process smaller batches
   - Ensure sufficient RAM and disk space

### Performance Tips

1. **Image Conversion**:
   - Use SSD storage for better I/O performance
   - Adjust worker count based on CPU cores
   - Process images in smaller batches for large collections

2. **Face Recognition**:
   - Use high-quality reference images
   - Reduce jitter samples for faster processing
   - Consider using 'hog' model for speed vs 'cnn' for accuracy

### Logging

Logs are stored in the `logs/` directory:
- `image_processor.log`: Main application log
- Console output shows real-time progress

To enable debug logging:
```bash
python src/image_processor_suite/main.py --log-level DEBUG convert /path/to/images
```

## Advanced Usage

### Batch Processing Scripts

Create custom scripts for repetitive tasks:

```python
#!/usr/bin/env python3
from pathlib import Path
import sys
sys.path.append('src')

from core.conversion.image_converter import ImageConverter

# Custom conversion script
converter = ImageConverter(quality=95, max_workers=8)
stats = converter.convert_images_batch(
    Path('/path/to/images'),
    recursive=True,
    delete_originals=False
)
print(f"Converted {stats['converted']} images")
```

### Integration with Other Tools

The core modules can be imported and used in other Python projects:

```python
from image_processor_suite.core.conversion import ImageConverter
from image_processor_suite.core.face_recognition import FaceSorter

# Use in your own applications
converter = ImageConverter()
sorter = FaceSorter(tolerance=0.4)
```

## Support

- **Documentation**: See `docs/` directory for detailed API documentation
- **Issues**: Report bugs and feature requests via GitHub Issues
- **Logs**: Check `logs/image_processor.log` for detailed error information

## License

This project is licensed under the MIT License. See `LICENSE` file for details.