#!/usr/bin/env python3
"""
Main GUI Interface Module

Consolidated GUI interface combining:
- Image conversion functionality
- Face recognition and sorting
- Unified progress tracking and error handling
- Modern tabbed interface with consistent styling
"""

import os
import sys
import threading
from pathlib import Path
from typing import Optional, Dict, Any, List
from dataclasses import dataclass

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from tkinter.scrolledtext import ScrolledText

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.image_utils import ImageUtils
from core.face_recognition_utils import FaceRecognitionUtils


@dataclass
class AppConfig:
    """Application configuration."""
    window_title: str = "Image Processor Suite"
    window_size: str = "1000x700"
    theme: str = "clam"
    log_max_lines: int = 1000
    

class ProgressDialog:
    """Modal progress dialog for long-running operations."""
    
    def __init__(self, parent, title: str = "Processing", cancelable: bool = True):
        self.parent = parent
        self.cancelable = cancelable
        self.cancelled = False
        
        # Create modal dialog
        self.dialog = tk.Toplevel(parent)
        self.dialog.title(title)
        self.dialog.geometry("400x150")
        self.dialog.resizable(False, False)
        self.dialog.transient(parent)
        self.dialog.grab_set()
        
        # Center dialog
        self.dialog.geometry("+%d+%d" % (
            parent.winfo_rootx() + 50,
            parent.winfo_rooty() + 50
        ))
        
        # Progress widgets
        self.progress_var = tk.StringVar(value="Initializing...")
        self.progress_label = ttk.Label(self.dialog, textvariable=self.progress_var)
        self.progress_label.pack(pady=10)
        
        self.progress_bar = ttk.Progressbar(self.dialog, mode='determinate')
        self.progress_bar.pack(pady=10, padx=20, fill='x')
        
        # Cancel button
        if cancelable:
            self.cancel_btn = ttk.Button(self.dialog, text="Cancel", command=self.cancel)
            self.cancel_btn.pack(pady=10)
        
        # Handle window close
        self.dialog.protocol("WM_DELETE_WINDOW", self.cancel)
    
    def update_progress(self, current: int, total: int, message: str = ""):
        """Update progress display."""
        if self.cancelled:
            return
        
        percentage = (current / max(total, 1)) * 100
        self.progress_bar['value'] = percentage
        
        if message:
            self.progress_var.set(f"{message} ({current}/{total})")
        else:
            self.progress_var.set(f"Processing... ({current}/{total})")
        
        self.dialog.update()
    
    def cancel(self):
        """Cancel the operation."""
        self.cancelled = True
        self.dialog.destroy()
    
    def close(self):
        """Close the dialog."""
        if not self.cancelled:
            self.dialog.destroy()


class LogHandler:
    """Custom log handler for GUI display."""
    
    def __init__(self, text_widget: ScrolledText, max_lines: int = 1000):
        self.text_widget = text_widget
        self.max_lines = max_lines
        self.line_count = 0
    
    def write(self, message: str):
        """Write message to text widget."""
        if not message.strip():
            return
        
        # Insert message
        self.text_widget.insert(tk.END, message + "\n")
        
        # Limit lines
        self.line_count += 1
        if self.line_count > self.max_lines:
            # Remove oldest lines
            lines_to_remove = self.line_count - self.max_lines
            self.text_widget.delete(1.0, f"{lines_to_remove + 1}.0")
            self.line_count = self.max_lines
        
        # Auto-scroll to bottom
        self.text_widget.see(tk.END)
        self.text_widget.update()
    
    def clear(self):
        """Clear all text."""
        self.text_widget.delete(1.0, tk.END)
        self.line_count = 0


class ImageConversionTab:
    """Image conversion functionality tab."""
    
    def __init__(self, parent_notebook, log_handler: LogHandler):
        self.log_handler = log_handler
        self.image_utils = ImageUtils()
        
        # Create tab frame
        self.frame = ttk.Frame(parent_notebook)
        parent_notebook.add(self.frame, text="Image Conversion")
        
        self._setup_ui()
    
    def _setup_ui(self):
        """Setup the conversion tab UI."""
        # Input section
        input_frame = ttk.LabelFrame(self.frame, text="Input", padding=10)
        input_frame.pack(fill='x', padx=10, pady=5)
        
        # Source directory
        ttk.Label(input_frame, text="Source Directory:").grid(row=0, column=0, sticky='w', pady=2)
        self.source_var = tk.StringVar()
        source_entry = ttk.Entry(input_frame, textvariable=self.source_var, width=50)
        source_entry.grid(row=0, column=1, padx=5, pady=2)
        ttk.Button(input_frame, text="Browse", command=self._browse_source).grid(row=0, column=2, padx=5, pady=2)
        
        # Output section
        output_frame = ttk.LabelFrame(self.frame, text="Output", padding=10)
        output_frame.pack(fill='x', padx=10, pady=5)
        
        # Output directory
        ttk.Label(output_frame, text="Output Directory:").grid(row=0, column=0, sticky='w', pady=2)
        self.output_var = tk.StringVar()
        output_entry = ttk.Entry(output_frame, textvariable=self.output_var, width=50)
        output_entry.grid(row=0, column=1, padx=5, pady=2)
        ttk.Button(output_frame, text="Browse", command=self._browse_output).grid(row=0, column=2, padx=5, pady=2)
        
        # Output format
        ttk.Label(output_frame, text="Output Format:").grid(row=1, column=0, sticky='w', pady=2)
        self.format_var = tk.StringVar(value="JPEG")
        format_combo = ttk.Combobox(output_frame, textvariable=self.format_var, 
                                   values=["JPEG", "PNG", "WEBP", "BMP", "TIFF"], 
                                   state="readonly", width=15)
        format_combo.grid(row=1, column=1, sticky='w', padx=5, pady=2)
        
        # Options section
        options_frame = ttk.LabelFrame(self.frame, text="Options", padding=10)
        options_frame.pack(fill='x', padx=10, pady=5)
        
        # Quality (for JPEG/WEBP)
        ttk.Label(options_frame, text="Quality (1-100):").grid(row=0, column=0, sticky='w', pady=2)
        self.quality_var = tk.IntVar(value=85)
        quality_scale = ttk.Scale(options_frame, from_=1, to=100, variable=self.quality_var, orient='horizontal')
        quality_scale.grid(row=0, column=1, sticky='ew', padx=5, pady=2)
        quality_label = ttk.Label(options_frame, textvariable=self.quality_var)
        quality_label.grid(row=0, column=2, padx=5, pady=2)
        
        # Recursive processing
        self.recursive_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(options_frame, text="Process subdirectories recursively", 
                       variable=self.recursive_var).grid(row=1, column=0, columnspan=3, sticky='w', pady=2)
        
        # Preserve structure
        self.preserve_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(options_frame, text="Preserve directory structure", 
                       variable=self.preserve_var).grid(row=2, column=0, columnspan=3, sticky='w', pady=2)
        
        # Configure grid weights
        options_frame.columnconfigure(1, weight=1)
        
        # Action buttons
        button_frame = ttk.Frame(self.frame)
        button_frame.pack(fill='x', padx=10, pady=10)
        
        ttk.Button(button_frame, text="Start Conversion", command=self._start_conversion).pack(side='left', padx=5)
        ttk.Button(button_frame, text="Clear Log", command=self.log_handler.clear).pack(side='right', padx=5)
    
    def _browse_source(self):
        """Browse for source directory."""
        directory = filedialog.askdirectory(title="Select Source Directory")
        if directory:
            self.source_var.set(directory)
    
    def _browse_output(self):
        """Browse for output directory."""
        directory = filedialog.askdirectory(title="Select Output Directory")
        if directory:
            self.output_var.set(directory)
    
    def _start_conversion(self):
        """Start image conversion process."""
        source_dir = self.source_var.get().strip()
        output_dir = self.output_var.get().strip()
        
        if not source_dir or not output_dir:
            messagebox.showerror("Error", "Please select both source and output directories.")
            return
        
        if not Path(source_dir).exists():
            messagebox.showerror("Error", "Source directory does not exist.")
            return
        
        # Prepare conversion options
        options = {
            'output_format': self.format_var.get(),
            'quality': self.quality_var.get(),
            'recursive': self.recursive_var.get(),
            'preserve_structure': self.preserve_var.get()
        }
        
        # Start conversion in separate thread
        thread = threading.Thread(
            target=self._run_conversion,
            args=(Path(source_dir), Path(output_dir), options),
            daemon=True
        )
        thread.start()
    
    def _run_conversion(self, source_dir: Path, output_dir: Path, options: Dict[str, Any]):
        """Run conversion process with progress tracking."""
        try:
            self.log_handler.write(f"Starting conversion: {source_dir} -> {output_dir}")
            self.log_handler.write(f"Options: {options}")
            
            # Create progress callback
            def progress_callback(current: int, total: int, filename: str):
                self.log_handler.write(f"Processing ({current}/{total}): {filename}")
            
            # Run batch conversion
            result = self.image_utils.batch_convert(
                source_dir=source_dir,
                output_dir=output_dir,
                output_format=options['output_format'],
                quality=options['quality'],
                recursive=options['recursive'],
                preserve_structure=options['preserve_structure'],
                progress_callback=progress_callback
            )
            
            # Display results
            if 'error' in result:
                self.log_handler.write(f"Conversion failed: {result['error']}")
            else:
                self.log_handler.write(f"Conversion completed successfully!")
                self.log_handler.write(f"Processed: {result['processed']} files")
                self.log_handler.write(f"Converted: {result['converted']} files")
                self.log_handler.write(f"Errors: {result['errors']} files")
                
                if result['error_details']:
                    self.log_handler.write("Error details:")
                    for error in result['error_details'][:10]:  # Show first 10 errors
                        self.log_handler.write(f"  {error}")
        
        except Exception as e:
            self.log_handler.write(f"Conversion error: {str(e)}")


class FaceRecognitionTab:
    """Face recognition functionality tab."""
    
    def __init__(self, parent_notebook, log_handler: LogHandler):
        self.log_handler = log_handler
        self.face_utils = FaceRecognitionUtils()
        
        # Create tab frame
        self.frame = ttk.Frame(parent_notebook)
        parent_notebook.add(self.frame, text="Face Recognition")
        
        self._setup_ui()
    
    def _setup_ui(self):
        """Setup the face recognition tab UI."""
        # Reference faces section
        ref_frame = ttk.LabelFrame(self.frame, text="Reference Faces", padding=10)
        ref_frame.pack(fill='x', padx=10, pady=5)
        
        # Add reference face
        ttk.Label(ref_frame, text="Reference Image:").grid(row=0, column=0, sticky='w', pady=2)
        self.ref_image_var = tk.StringVar()
        ref_entry = ttk.Entry(ref_frame, textvariable=self.ref_image_var, width=40)
        ref_entry.grid(row=0, column=1, padx=5, pady=2)
        ttk.Button(ref_frame, text="Browse", command=self._browse_reference).grid(row=0, column=2, padx=5, pady=2)
        
        ttk.Label(ref_frame, text="Person Name:").grid(row=1, column=0, sticky='w', pady=2)
        self.ref_name_var = tk.StringVar()
        ttk.Entry(ref_frame, textvariable=self.ref_name_var, width=20).grid(row=1, column=1, sticky='w', padx=5, pady=2)
        
        ttk.Label(ref_frame, text="Destination Folder:").grid(row=2, column=0, sticky='w', pady=2)
        self.ref_dest_var = tk.StringVar()
        dest_entry = ttk.Entry(ref_frame, textvariable=self.ref_dest_var, width=40)
        dest_entry.grid(row=2, column=1, padx=5, pady=2)
        ttk.Button(ref_frame, text="Browse", command=self._browse_destination).grid(row=2, column=2, padx=5, pady=2)
        
        ttk.Button(ref_frame, text="Add Reference Face", command=self._add_reference).grid(row=3, column=1, pady=10)
        
        # Reference faces list
        list_frame = ttk.Frame(ref_frame)
        list_frame.grid(row=4, column=0, columnspan=3, sticky='ew', pady=5)
        
        self.ref_listbox = tk.Listbox(list_frame, height=4)
        self.ref_listbox.pack(side='left', fill='both', expand=True)
        
        scrollbar = ttk.Scrollbar(list_frame, orient='vertical', command=self.ref_listbox.yview)
        scrollbar.pack(side='right', fill='y')
        self.ref_listbox.config(yscrollcommand=scrollbar.set)
        
        ttk.Button(ref_frame, text="Clear All References", command=self._clear_references).grid(row=5, column=1, pady=5)
        
        # Processing section
        process_frame = ttk.LabelFrame(self.frame, text="Image Processing", padding=10)
        process_frame.pack(fill='x', padx=10, pady=5)
        
        # Source directory
        ttk.Label(process_frame, text="Source Directory:").grid(row=0, column=0, sticky='w', pady=2)
        self.source_var = tk.StringVar()
        source_entry = ttk.Entry(process_frame, textvariable=self.source_var, width=50)
        source_entry.grid(row=0, column=1, padx=5, pady=2)
        ttk.Button(process_frame, text="Browse", command=self._browse_source).grid(row=0, column=2, padx=5, pady=2)
        
        # Options
        self.recursive_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(process_frame, text="Process subdirectories recursively", 
                       variable=self.recursive_var).grid(row=1, column=0, columnspan=3, sticky='w', pady=2)
        
        # Tolerance setting
        ttk.Label(process_frame, text="Matching Tolerance (0.0-1.0):").grid(row=2, column=0, sticky='w', pady=2)
        self.tolerance_var = tk.DoubleVar(value=0.5)
        tolerance_scale = ttk.Scale(process_frame, from_=0.0, to=1.0, variable=self.tolerance_var, orient='horizontal')
        tolerance_scale.grid(row=2, column=1, sticky='ew', padx=5, pady=2)
        tolerance_label = ttk.Label(process_frame, textvariable=self.tolerance_var)
        tolerance_label.grid(row=2, column=2, padx=5, pady=2)
        
        # Configure grid weights
        ref_frame.columnconfigure(1, weight=1)
        process_frame.columnconfigure(1, weight=1)
        
        # Action buttons
        button_frame = ttk.Frame(self.frame)
        button_frame.pack(fill='x', padx=10, pady=10)
        
        ttk.Button(button_frame, text="Start Face Sorting", command=self._start_sorting).pack(side='left', padx=5)
        ttk.Button(button_frame, text="Clear Log", command=self.log_handler.clear).pack(side='right', padx=5)
    
    def _browse_reference(self):
        """Browse for reference image."""
        filename = filedialog.askopenfilename(
            title="Select Reference Image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff *.webp")]
        )
        if filename:
            self.ref_image_var.set(filename)
            # Auto-fill name from filename
            if not self.ref_name_var.get():
                self.ref_name_var.set(Path(filename).stem)
    
    def _browse_destination(self):
        """Browse for destination directory."""
        directory = filedialog.askdirectory(title="Select Destination Directory")
        if directory:
            self.ref_dest_var.set(directory)
    
    def _browse_source(self):
        """Browse for source directory."""
        directory = filedialog.askdirectory(title="Select Source Directory")
        if directory:
            self.source_var.set(directory)
    
    def _add_reference(self):
        """Add reference face."""
        image_path = self.ref_image_var.get().strip()
        name = self.ref_name_var.get().strip()
        dest_path = self.ref_dest_var.get().strip()
        
        if not all([image_path, name, dest_path]):
            messagebox.showerror("Error", "Please fill in all reference face fields.")
            return
        
        if not Path(image_path).exists():
            messagebox.showerror("Error", "Reference image file does not exist.")
            return
        
        # Update tolerance
        self.face_utils.set_tolerance(self.tolerance_var.get())
        
        # Add reference face
        success = self.face_utils.add_reference_face(
            Path(image_path), Path(dest_path), name
        )
        
        if success:
            self.ref_listbox.insert(tk.END, f"{name} -> {dest_path}")
            self.log_handler.write(f"Added reference face: {name}")
            
            # Clear fields
            self.ref_image_var.set("")
            self.ref_name_var.set("")
            self.ref_dest_var.set("")
        else:
            messagebox.showerror("Error", "Failed to add reference face. Check the log for details.")
    
    def _clear_references(self):
        """Clear all reference faces."""
        self.face_utils.clear_reference_faces()
        self.ref_listbox.delete(0, tk.END)
        self.log_handler.write("Cleared all reference faces")
    
    def _start_sorting(self):
        """Start face sorting process."""
        source_dir = self.source_var.get().strip()
        
        if not source_dir:
            messagebox.showerror("Error", "Please select a source directory.")
            return
        
        if not Path(source_dir).exists():
            messagebox.showerror("Error", "Source directory does not exist.")
            return
        
        if not self.face_utils.reference_faces:
            messagebox.showerror("Error", "Please add at least one reference face.")
            return
        
        # Update tolerance
        self.face_utils.set_tolerance(self.tolerance_var.get())
        
        # Start sorting in separate thread
        thread = threading.Thread(
            target=self._run_sorting,
            args=(Path(source_dir),),
            daemon=True
        )
        thread.start()
    
    def _run_sorting(self, source_dir: Path):
        """Run face sorting process with progress tracking."""
        try:
            self.log_handler.write(f"Starting face sorting: {source_dir}")
            self.log_handler.write(f"Reference faces: {len(self.face_utils.reference_faces)}")
            self.log_handler.write(f"Tolerance: {self.face_utils.tolerance}")
            
            # Create progress callback
            def progress_callback(current: int, total: int, filename: str):
                self.log_handler.write(f"Processing ({current}/{total}): {Path(filename).name}")
            
            # Run face sorting
            result = self.face_utils.sort_images(
                source_dir=source_dir,
                recursive=self.recursive_var.get(),
                progress_callback=progress_callback
            )
            
            # Display results
            if 'error' in result:
                self.log_handler.write(f"Face sorting failed: {result['error']}")
            else:
                self.log_handler.write(f"Face sorting completed successfully!")
                self.log_handler.write(f"Processed: {result['processed']} files")
                self.log_handler.write(f"Matched: {result['matched']} files")
                self.log_handler.write(f"Unmatched: {result['unmatched']} files")
                self.log_handler.write(f"Errors: {result['errors']} files")
                
                if result['moved_files']:
                    self.log_handler.write("Files moved to:")
                    for dest_dir, count in result['moved_files'].items():
                        self.log_handler.write(f"  {dest_dir}: {count} files")
                
                if result['error_details']:
                    self.log_handler.write("Error details:")
                    for error in result['error_details'][:10]:  # Show first 10 errors
                        self.log_handler.write(f"  {error}")
        
        except Exception as e:
            self.log_handler.write(f"Face sorting error: {str(e)}")


class MainApplication:
    """Main application class."""
    
    def __init__(self, config: AppConfig = None):
        self.config = config or AppConfig()
        
        # Create main window
        self.root = tk.Tk()
        self.root.title(self.config.window_title)
        self.root.geometry(self.config.window_size)
        
        # Set theme
        style = ttk.Style()
        style.theme_use(self.config.theme)
        
        self._setup_ui()
        
        # Handle window close
        self.root.protocol("WM_DELETE_WINDOW", self._on_closing)
    
    def _setup_ui(self):
        """Setup the main UI."""
        # Create main paned window
        main_paned = ttk.PanedWindow(self.root, orient='vertical')
        main_paned.pack(fill='both', expand=True, padx=5, pady=5)
        
        # Create notebook for tabs
        self.notebook = ttk.Notebook(main_paned)
        
        # Create log frame
        log_frame = ttk.LabelFrame(main_paned, text="Log Output", padding=5)
        
        # Setup log text widget
        self.log_text = ScrolledText(log_frame, height=10, wrap='word')
        self.log_text.pack(fill='both', expand=True)
        
        # Create log handler
        self.log_handler = LogHandler(self.log_text, self.config.log_max_lines)
        
        # Create tabs
        self.conversion_tab = ImageConversionTab(self.notebook, self.log_handler)
        self.face_recognition_tab = FaceRecognitionTab(self.notebook, self.log_handler)
        
        # Add to paned window
        main_paned.add(self.notebook, weight=3)
        main_paned.add(log_frame, weight=1)
        
        # Initial log message
        self.log_handler.write("Image Processor Suite initialized")
        self.log_handler.write("Select a tab above to begin processing")
    
    def _on_closing(self):
        """Handle application closing."""
        # Cancel any ongoing operations
        if hasattr(self.face_recognition_tab, 'face_utils'):
            self.face_recognition_tab.face_utils.cancel_processing()
        
        self.root.destroy()
    
    def run(self):
        """Start the application."""
        self.root.mainloop()


def main():
    """Main entry point."""
    try:
        app = MainApplication()
        app.run()
    except Exception as e:
        messagebox.showerror("Error", f"Application failed to start: {str(e)}")


if __name__ == "__main__":
    main()