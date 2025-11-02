#!/usr/bin/env python3
"""
Main Entry Point for Image Processor Suite

Provides command-line and GUI entry points for the application.
"""

import sys
import argparse
from pathlib import Path
from typing import Optional
import logging

# Add src to path for imports
src_path = Path(__file__).parent.parent
sys.path.insert(0, str(src_path))

try:
    from gui.main_interface import MainApplication
except ImportError:
    MainApplication = None  # GUI not available

from core.image_utils import ImageUtils
from core.face_recognition_utils import FaceRecognitionUtils


def setup_logging(log_level: str = 'INFO') -> logging.Logger:
    """
    Setup application logging.
    
    Args:
        log_level: Logging level
        
    Returns:
        Configured logger
    """
    # Create logs directory
    log_dir = Path('logs')
    log_dir.mkdir(exist_ok=True)
    
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_dir / 'image_processor.log', encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    return logging.getLogger('image_processor_suite')


def main_gui() -> None:
    """
    Launch the GUI application.
    """
    logger = setup_logging()
    logger.info("Starting Image Processor Suite GUI")
    
    if MainApplication is None:
        logger.error("GUI not available - tkinter may not be installed")
        print("Error: GUI not available. Please install tkinter or use CLI mode.")
        sys.exit(1)
    
    try:
        app = MainApplication()
        app.run()
    except Exception as e:
        logger.error(f"GUI application error: {e}")
        sys.exit(1)


def main_cli() -> None:
    """
    Run command-line interface.
    """
    parser = argparse.ArgumentParser(
        description='Image Processor Suite - Convert images and sort by faces'
    )
    
    # Global options
    parser.add_argument('--log-level', default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Logging level')
    
    # Subcommands
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Convert command
    convert_parser = subparsers.add_parser('convert', help='Convert images')
    convert_parser.add_argument('source', type=Path, help='Source directory')
    convert_parser.add_argument('--output', type=Path, help='Output directory')
    convert_parser.add_argument('--quality', type=int, default=95,
                               help='JPEG quality (1-100)')
    convert_parser.add_argument('--recursive', action='store_true',
                               help='Process subdirectories')
    convert_parser.add_argument('--keep-originals', action='store_true',
                               help='Keep original files')
    convert_parser.add_argument('--workers', type=int, default=4,
                               help='Number of worker threads')
    
    # Sort command
    sort_parser = subparsers.add_parser('sort', help='Sort images by faces')
    sort_parser.add_argument('source', type=Path, help='Source directory')
    sort_parser.add_argument('--face', type=Path, action='append', required=True,
                            help='Reference face image (can be used multiple times)')
    sort_parser.add_argument('--dest', type=Path, action='append', required=True,
                            help='Destination directory (one per face)')
    sort_parser.add_argument('--tolerance', type=float, default=0.5,
                            help='Face matching tolerance')
    sort_parser.add_argument('--recursive', action='store_true',
                            help='Process subdirectories')
    sort_parser.add_argument('--workers', type=int,
                            help='Number of worker processes')
    sort_parser.add_argument('--jitter', type=int, default=1,
                            help='Face encoding jitter samples')
    
    # GUI command
    gui_parser = subparsers.add_parser('gui', help='Launch GUI application')
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging(args.log_level)
    
    if args.command == 'convert':
        run_convert_command(args, logger)
    elif args.command == 'sort':
        run_sort_command(args, logger)
    elif args.command == 'gui':
        main_gui()
    else:
        # Default to GUI if no command specified
        main_gui()


def run_convert_command(args, logger: logging.Logger) -> None:
    """
    Run image conversion command.
    
    Args:
        args: Command line arguments
        logger: Logger instance
    """
    try:
        # Validate source directory
        if not args.source.exists() or not args.source.is_dir():
            logger.error(f"Source directory does not exist: {args.source}")
            sys.exit(1)
        
        # Setup output directory
        output_dir = args.output or args.source / 'converted'
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create converter
        converter = ImageUtils(
            quality=args.quality,
            max_workers=args.workers,
            logger=logger
        )
        
        logger.info(f"Starting conversion: {args.source} -> {output_dir}")
        logger.info(f"Quality: {args.quality}, Workers: {args.workers}")
        logger.info(f"Recursive: {args.recursive}, Keep originals: {args.keep_originals}")
        
        # Run conversion
        stats = converter.convert_images_batch(
            args.source,
            recursive=args.recursive,
            delete_originals=not args.keep_originals
        )
        
        # Print results
        print("\n=== CONVERSION RESULTS ===")
        print(f"Total files: {stats['total_files']}")
        print(f"Converted: {stats['converted']}")
        print(f"Failed: {stats['failed']}")
        print(f"Deleted originals: {stats['deleted']}")
        
        if stats['errors']:
            print(f"\nErrors ({len(stats['errors'])}):")
            for error in stats['errors'][:10]:  # Show first 10 errors
                print(f"  {error}")
            if len(stats['errors']) > 10:
                print(f"  ... and {len(stats['errors']) - 10} more")
        
        logger.info("Conversion completed successfully")
        
    except Exception as e:
        logger.error(f"Conversion failed: {e}")
        sys.exit(1)


def run_sort_command(args, logger: logging.Logger) -> None:
    """
    Run face recognition sorting command.
    
    Args:
        args: Command line arguments
        logger: Logger instance
    """
    try:
        # Validate arguments
        if not args.source.exists() or not args.source.is_dir():
            logger.error(f"Source directory does not exist: {args.source}")
            sys.exit(1)
        
        if len(args.face) != len(args.dest):
            logger.error("Number of face images must match number of destinations")
            sys.exit(1)
        
        # Validate face images
        for face_path in args.face:
            if not face_path.exists() or not face_path.is_file():
                logger.error(f"Face image does not exist: {face_path}")
                sys.exit(1)
        
        # Create sorter
        sorter = FaceRecognitionUtils(
            tolerance=args.tolerance,
            jitter=args.jitter,
            max_workers=args.workers,
            logger=logger
        )
        
        # Add reference faces
        for face_path, dest_path in zip(args.face, args.dest):
            success = sorter.add_reference_face(face_path, dest_path)
            if not success:
                logger.error(f"Failed to add reference face: {face_path}")
                sys.exit(1)
        
        logger.info(f"Starting face recognition sorting: {args.source}")
        logger.info(f"Tolerance: {args.tolerance}, Jitter: {args.jitter}")
        logger.info(f"Workers: {args.workers}, Recursive: {args.recursive}")
        logger.info(f"Reference faces: {len(args.face)}")
        
        # Run sorting
        stats = sorter.sort_images(args.source, recursive=args.recursive)
        
        # Print results
        print("\n=== SORTING RESULTS ===")
        print(f"Total images: {stats['total_images']}")
        print(f"Moved: {stats['moved']}")
        print(f"Unmatched: {stats['unmatched']}")
        print(f"Errors: {stats['errors']}")
        
        print("\nDestination breakdown:")
        for dest, count in stats['destinations'].items():
            print(f"  {Path(dest).name}: {count} images")
        
        if stats['error_details']:
            print(f"\nError details ({len(stats['error_details'])}):")
            for error in stats['error_details'][:10]:  # Show first 10 errors
                print(f"  {error}")
            if len(stats['error_details']) > 10:
                print(f"  ... and {len(stats['error_details']) - 10} more")
        
        logger.info("Face recognition sorting completed successfully")
        
    except Exception as e:
        logger.error(f"Face recognition sorting failed: {e}")
        sys.exit(1)


def main() -> None:
    """
    Main entry point.
    """
    # Check if GUI should be launched (no command line arguments)
    if len(sys.argv) == 1:
        main_gui()
    else:
        main_cli()


if __name__ == '__main__':
    main()