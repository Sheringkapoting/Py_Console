# Image Processor Suite

A comprehensive image processing application that combines WEBP to JPG conversion and face recognition-based sorting into a single, user-friendly interface.

## Features

### 🖼️ Image Conversion
- **WEBP to JPG Conversion**: Batch convert WEBP, PNG, and JPEG images to JPG format
- **Recursive Processing**: Process entire directory trees
- **Progress Tracking**: Real-time progress bars with detailed statistics
- **Safe Operations**: Automatic backup and validation
- **Memory Efficient**: Handles large images with built-in safety limits

### 👤 Face Recognition & Sorting
- **Secure Face Recognition**: Advanced face detection and matching
- **Batch Processing**: Sort thousands of images efficiently
- **Multiple Reference Faces**: Support for multiple face templates
- **Customizable Tolerance**: Adjustable matching sensitivity
- **Safe File Operations**: Unique naming and traversal protection
- **Parallel Processing**: Multi-threaded analysis for speed

### 🖥️ User Interface
- **Professional GUI**: Modern, intuitive interface built with tkinter
- **Integrated Workflow**: Seamless switching between conversion and sorting
- **Progress Monitoring**: Real-time feedback with detailed progress bars
- **Error Handling**: Comprehensive error reporting and recovery
- **Configuration Management**: Persistent settings and preferences

## Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Quick Install
```bash
# Clone the repository
git clone https://github.com/imageprocessor/image-processor-suite.git
cd image-processor-suite

# Install dependencies
pip install -r requirements.txt

# Install the package
pip install -e .
```

### Development Install
```bash
# Install with development dependencies
pip install -r requirements-dev.txt
pip install -e .
```

## Usage

### GUI Application
```bash
# Launch the GUI application
image-processor-gui

# Or run directly
python -m image_processor_suite.main --gui
```

### Command Line Interface
```bash
# Image conversion
image-processor convert --source /path/to/images --recursive

# Face recognition sorting
image-processor sort --source /path/to/images --face /path/to/reference.jpg --dest /path/to/sorted

# Show help
image-processor --help
```

## Project Structure

```
image_processor_suite/
├── src/
│   ├── image_processor_suite/     # Main package
│   ├── core/                      # Core functionality
│   │   ├── conversion/            # Image conversion modules
│   │   └── face_recognition/      # Face recognition modules
│   ├── gui/                       # GUI components
│   │   ├── components/            # Reusable UI components
│   │   └── dialogs/               # Dialog windows
│   └── utils/                     # Utility modules
│       ├── validators/            # Input validation
│       └── helpers/               # Helper functions
├── tests/                         # Test suite
├── docs/                          # Documentation
├── config/                        # Configuration files
├── assets/                        # Static assets
└── logs/                          # Application logs
```

## Configuration

The application uses a configuration file located at `config/settings.json`. Key settings include:

- **Image Processing**: File size limits, supported formats, quality settings
- **Face Recognition**: Tolerance levels, processing threads, model settings
- **GUI**: Theme preferences, window sizes, default directories
- **Logging**: Log levels, file rotation, output formats

## Safety Features

- **File Validation**: Comprehensive image validation before processing
- **Memory Protection**: Built-in limits to prevent memory exhaustion
- **Backup Creation**: Automatic backup of original files (optional)
- **Path Traversal Protection**: Prevents writing outside designated directories
- **Error Recovery**: Graceful handling of corrupted or invalid files

## Performance

- **Multi-threading**: Parallel processing for face recognition
- **Memory Optimization**: Efficient handling of large image collections
- **Progress Tracking**: Real-time feedback on processing status
- **Batch Operations**: Optimized for processing thousands of images

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Setup

```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
pytest

# Run linting
flake8 src/
black src/
mypy src/

# Build documentation
cd docs/
make html
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Support

- **Documentation**: [Full documentation](docs/)
- **Issues**: [GitHub Issues](https://github.com/imageprocessor/image-processor-suite/issues)
- **Discussions**: [GitHub Discussions](https://github.com/imageprocessor/image-processor-suite/discussions)

## Changelog

See [CHANGELOG.md](CHANGELOG.md) for a detailed history of changes.

## Acknowledgments

- Built with [Pillow](https://pillow.readthedocs.io/) for image processing
- Uses [face_recognition](https://github.com/ageitgey/face_recognition) for face detection
- GUI built with Python's built-in [tkinter](https://docs.python.org/3/library/tkinter.html)
- Progress bars powered by [tqdm](https://tqdm.github.io/)