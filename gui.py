"""
96-Well Plate Barcode Scanner GUI
Tkinter interface with live camera preview and focus-stack style capture workflow.
"""

import os
import tempfile
import threading
import time
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext, ttk

import cv2
from PIL import Image, ImageTk

from main import BarcodeScanner


class BarcodeScannerGUI:
    """GUI for the barcode scanner application."""

    PREVIEW_SIZE = (1920, 1080)

    def __init__(self, root):
        self.root = root
        self.root.title("96-Well Plate Barcode Scanner")
        self.root.geometry("2400x1200")
        self.root.minsize(1100, 750)

        self.scanner = BarcodeScanner()
        self.current_image_path = None
        self.current_results = {}

        self.capture = None
        self.preview_running = False
        self.scan_thread = None
        self.preview_job = None
        self.current_frame = None
        self.preview_photo = None
        self.current_display_image = None
        self.last_stack_metadata = []

        # Track original image for dynamic resizing
        self._original_pil_image = None
        self._resize_job = None

        self.setup_ui()
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def setup_ui(self):
        """Setup the user interface."""
        self.main_frame = ttk.Frame(self.root, padding="10")
        self.main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        self.main_frame.columnconfigure(1, weight=3)
        self.main_frame.columnconfigure(2, weight=2)
        self.main_frame.rowconfigure(1, weight=1)

        title_label = ttk.Label(
            self.main_frame,
            text="96-Well Plate Barcode Scanner",
            font=("Helvetica", 18, "bold"),
        )
        title_label.grid(row=0, column=0, columnspan=3, pady=(0, 10))

        self.setup_control_panel()
        self.setup_image_panel()
        self.setup_results_panel()

    def setup_control_panel(self):
        """Setup the left control panel."""
        # Use a canvas and scrollbar for the control panel in case it gets too tall
        control_canvas = tk.Canvas(self.main_frame, width=280, highlightthickness=0)
        scrollbar = ttk.Scrollbar(self.main_frame, orient="vertical", command=control_canvas.yview)

        control_frame = ttk.LabelFrame(control_canvas, text="Controls", padding="10")

        control_canvas.configure(yscrollcommand=scrollbar.set)

        control_canvas.grid(row=1, column=0, sticky=(tk.N, tk.S, tk.W, tk.E), padx=(0, 10))
        scrollbar.grid(row=1, column=0, sticky=(tk.N, tk.S, tk.E))

        control_canvas.create_window((0, 0), window=control_frame, anchor="nw", width=260)

        def on_configure(event):
            control_canvas.configure(scrollregion=control_canvas.bbox("all"))

        control_frame.bind("<Configure>", on_configure)

        control_frame.columnconfigure(0, weight=1)

        self.start_camera_btn = ttk.Button(
            control_frame,
            text="Start Camera Preview",
            command=self.start_camera_preview,
        )
        self.start_camera_btn.grid(row=0, column=0, pady=5, sticky=(tk.W, tk.E))

        self.stop_camera_btn = ttk.Button(
            control_frame,
            text="Stop Camera Preview",
            command=self.stop_camera_preview,
            state="disabled",
        )
        self.stop_camera_btn.grid(row=1, column=0, pady=5, sticky=(tk.W, tk.E))

        self.capture_scan_btn = ttk.Button(
            control_frame,
            text="Capture + Scan",
            command=self.start_live_scan,
            state="disabled",
        )
        self.capture_scan_btn.grid(row=2, column=0, pady=5, sticky=(tk.W, tk.E))

        self.capture_only_btn = ttk.Button(
            control_frame,
            text="Capture Only (Save)",
            command=self.capture_only_save,
            state="disabled",
        )
        self.capture_only_btn.grid(row=3, column=0, pady=5, sticky=(tk.W, tk.E))

        self.select_btn = ttk.Button(
            control_frame,
            text="Select Image",
            command=self.select_image,
        )
        self.select_btn.grid(row=4, column=0, pady=5, sticky=(tk.W, tk.E))

        self.scan_btn = ttk.Button(
            control_frame,
            text="Scan Loaded Image",
            command=self.scan_plate,
            state="disabled",
        )
        self.scan_btn.grid(row=5, column=0, pady=5, sticky=(tk.W, tk.E))

        self.scan_sweep_btn = ttk.Button(
            control_frame,
            text="Scan Sweep Files (Streaming)",
            command=self.scan_sweep_files,
        )
        self.scan_sweep_btn.grid(row=6, column=0, pady=5, sticky=(tk.W, tk.E))

        ttk.Separator(control_frame, orient="horizontal").grid(
            row=6, column=0, pady=10, sticky=(tk.W, tk.E)
        )

        # --- MANUAL FOCUS CONTROL ---
        ttk.Label(control_frame, text="Manual Focus:").grid(row=7, column=0, pady=(5, 0), sticky=tk.W)
        self.manual_focus_var = tk.IntVar(value=400)
        self.manual_focus_scale = ttk.Scale(
            control_frame,
            from_=0,
            to=1024,
            variable=self.manual_focus_var,
            orient="horizontal",
            command=self.on_manual_focus_change
        )
        self.manual_focus_scale.grid(row=8, column=0, pady=5, sticky=(tk.W, tk.E))
        self.manual_focus_spinbox = tk.Spinbox(
            control_frame,
            from_=0,
            to=1024,
            textvariable=self.manual_focus_var,
            width=8,
            command=self.on_manual_focus_spinbox_change
        )
        self.manual_focus_spinbox.grid(row=9, column=0, sticky=tk.W)
        self.manual_focus_spinbox.bind('<Return>', self.on_manual_focus_spinbox_return)
        self.manual_focus_spinbox.bind('<FocusOut>', self.on_manual_focus_spinbox_return)

        # --- MANUAL EXPOSURE CONTROL ---
        ttk.Label(control_frame, text="Manual Exposure:").grid(row=10, column=0, pady=(10, 0), sticky=tk.W)
        self.manual_exposure_var = tk.IntVar(value=-4)
        self.manual_exposure_scale = ttk.Scale(
            control_frame,
            from_=-13,
            to=-1,
            variable=self.manual_exposure_var,
            orient="horizontal",
            command=self.on_manual_exposure_change
        )
        self.manual_exposure_scale.grid(row=11, column=0, pady=5, sticky=(tk.W, tk.E))
        self.manual_exposure_spinbox = tk.Spinbox(
            control_frame,
            from_=-13,
            to=-1,
            textvariable=self.manual_exposure_var,
            width=8,
            command=self.on_manual_exposure_spinbox_change
        )
        self.manual_exposure_spinbox.grid(row=12, column=0, sticky=tk.W)
        self.manual_exposure_spinbox.bind('<Return>', self.on_manual_exposure_spinbox_return)
        self.manual_exposure_spinbox.bind('<FocusOut>', self.on_manual_exposure_spinbox_return)
        # ----------------------------

        # --- NEW HARDWARE FOCUS CONTROLS ---
        ttk.Label(control_frame, text="Focus Sweep Start (e.g. 40):").grid(row=13, column=0, pady=(10, 0), sticky=tk.W)
        self.focus_start_var = tk.IntVar(value=40)
        ttk.Spinbox(control_frame, from_=0, to=1024, textvariable=self.focus_start_var, width=6).grid(
            row=14, column=0, pady=5, sticky=tk.W
        )

        ttk.Label(control_frame, text="Focus Sweep End (e.g. 100):").grid(row=15, column=0, pady=(10, 0), sticky=tk.W)
        self.focus_end_var = tk.IntVar(value=100)
        ttk.Spinbox(control_frame, from_=0, to=1024, textvariable=self.focus_end_var, width=6).grid(
            row=16, column=0, pady=5, sticky=tk.W
        )

        ttk.Label(control_frame, text="Sweep frames (steps):").grid(row=17, column=0, pady=(10, 0), sticky=tk.W)
        self.sweep_count_var = tk.IntVar(value=7)
        ttk.Spinbox(control_frame, from_=1, to=25, increment=1, textvariable=self.sweep_count_var, width=6).grid(
            row=18, column=0, pady=5, sticky=tk.W
        )

        ttk.Label(control_frame, text="Motor Settle Delay (ms):").grid(row=19, column=0, pady=(10, 0), sticky=tk.W)
        self.capture_delay_var = tk.IntVar(value=500)
        ttk.Spinbox(control_frame, from_=50, to=2000, increment=50, textvariable=self.capture_delay_var, width=8).grid(
            row=20, column=0, pady=5, sticky=tk.W
        )
        # -----------------------------------

        self.distortion_var = tk.BooleanVar(value=False)
        self.distortion_check = ttk.Checkbutton(
            control_frame,
            text="Apply Distortion Correction",
            variable=self.distortion_var,
        )
        self.distortion_check.grid(row=21, column=0, pady=5, sticky=tk.W)

        ttk.Label(control_frame, text="k1 (distortion):").grid(row=22, column=0, pady=(10, 0), sticky=tk.W)
        self.k1_var = tk.DoubleVar(value=-0.15)
        self.k1_scale = ttk.Scale(
            control_frame,
            from_=-1.0,
            to=1.0,
            variable=self.k1_var,
            orient="horizontal",
        )
        self.k1_scale.grid(row=23, column=0, pady=5, sticky=(tk.W, tk.E))
        self.k1_label = ttk.Label(control_frame, text="-0.15")
        self.k1_label.grid(row=24, column=0, sticky=tk.W)
        self.k1_scale.configure(command=lambda v: self.k1_label.configure(text=f"{float(v):.2f}"))

        ttk.Label(control_frame, text="k2 (secondary):").grid(row=25, column=0, pady=(10, 0), sticky=tk.W)
        self.k2_var = tk.DoubleVar(value=0.05)
        self.k2_scale = ttk.Scale(
            control_frame,
            from_=-0.5,
            to=0.5,
            variable=self.k2_var,
            orient="horizontal",
        )
        self.k2_scale.grid(row=26, column=0, pady=5, sticky=(tk.W, tk.E))
        self.k2_label = ttk.Label(control_frame, text="0.05")
        self.k2_label.grid(row=27, column=0, sticky=tk.W)
        self.k2_scale.configure(command=lambda v: self.k2_label.configure(text=f"{float(v):.2f}"))

        ttk.Separator(control_frame, orient="horizontal").grid(
            row=28, column=0, pady=10, sticky=(tk.W, tk.E))

        self.progress_var = tk.DoubleVar(value=0)
        self.progress = ttk.Progressbar(control_frame, variable=self.progress_var, maximum=100)
        self.progress.grid(row=29, column=0, pady=5, sticky=(tk.W, tk.E))

        self.phase_var = tk.StringVar(value="Idle")
        ttk.Label(control_frame, text="Current phase:").grid(row=30, column=0, sticky=tk.W)
        ttk.Label(control_frame, textvariable=self.phase_var, foreground="darkgreen").grid(
            row=31, column=0, pady=3, sticky=tk.W
        )

        self.status_var = tk.StringVar(value="Ready")
        ttk.Label(control_frame, text="Status:").grid(row=32, column=0, sticky=tk.W)
        ttk.Label(control_frame, textvariable=self.status_var, foreground="blue", wraplength=240).grid(
            row=33, column=0, pady=3, sticky=tk.W
        )

        self.export_txt_btn = ttk.Button(
            control_frame,
            text="Export to TXT",
            command=lambda: self.export_results("txt"),
            state="disabled",
        )
        self.export_txt_btn.grid(row=34, column=0, pady=(10, 5), sticky=(tk.W, tk.E))

        self.export_csv_btn = ttk.Button(
            control_frame,
            text="Export to CSV",
            command=lambda: self.export_results("csv"),
            state="disabled",
        )
        self.export_csv_btn.grid(row=35, column=0, pady=5, sticky=(tk.W, tk.E))

    def setup_image_panel(self):
        """Setup the middle image preview panel."""
        image_frame = ttk.LabelFrame(self.main_frame, text="Live Preview / Selected Image", padding="10")
        image_frame.grid(row=1, column=1, sticky=(tk.N, tk.S, tk.E, tk.W))
        image_frame.columnconfigure(0, weight=1)
        image_frame.rowconfigure(0, weight=1)

        self.canvas = tk.Canvas(image_frame, bg="gray15", highlightthickness=0)
        self.canvas.grid(row=0, column=0, sticky=(tk.N, tk.S, tk.E, tk.W))

        self.canvas.create_text(
            350,
            220,
            text="Start camera preview or select an image",
            fill="white",
            font=("Helvetica", 13),
        )

    def setup_results_panel(self):
        """Setup the right results panel."""
        results_frame = ttk.LabelFrame(self.main_frame, text="Scan Results", padding="10")
        results_frame.grid(row=1, column=2, sticky=(tk.N, tk.S, tk.E, tk.W), padx=(10, 0))
        results_frame.columnconfigure(0, weight=1)
        results_frame.rowconfigure(3, weight=1)

        grid_frame = ttk.Frame(results_frame)
        grid_frame.grid(row=0, column=0, pady=(0, 10))

        self.grid_cells = {}

        ttk.Label(grid_frame, text="").grid(row=0, column=0, padx=2, pady=2)
        for col in range(12, 0, -1):
            ttk.Label(grid_frame, text=str(col), width=3).grid(row=0, column=13 - col, padx=2, pady=2)

        for row_idx, row_label in enumerate(self.scanner.ROWS):
            ttk.Label(grid_frame, text=row_label).grid(row=row_idx + 1, column=0, padx=2, pady=2)
            for col in range(12, 0, -1):
                cell = tk.Label(
                    grid_frame,
                    text="",
                    width=3,
                    bg="lightgray",
                    relief="solid",
                    borderwidth=1,
                )
                cell.grid(row=row_idx + 1, column=13 - col, padx=1, pady=1)
                well_id = f"{row_label}{col}"
                self.grid_cells[well_id] = cell

        self.stats_var = tk.StringVar(value="Detected: 0/96")
        ttk.Label(results_frame, textvariable=self.stats_var, font=("Helvetica", 10, "bold")).grid(
            row=1, column=0, pady=5, sticky=tk.W
        )

        self.capture_info_var = tk.StringVar(
            value="Live mode sweeps hardware focus and keeps the sharpest barcode result for each well."
        )
        ttk.Label(results_frame, textvariable=self.capture_info_var, wraplength=360, foreground="gray25").grid(
            row=2, column=0, sticky=(tk.W, tk.E), pady=(0, 10)
        )

        self.results_text = scrolledtext.ScrolledText(
            results_frame,
            width=42,
            height=22,
            wrap=tk.WORD,
        )
        self.results_text.grid(row=3, column=0, sticky=(tk.N, tk.S, tk.E, tk.W))

    def on_manual_focus_change(self, value):
        """Apply focus to the camera if running."""
        focus_val = int(float(value))
        if self.preview_running and self.capture:
            self.capture.set(cv2.CAP_PROP_FOCUS, focus_val)

    def on_manual_focus_spinbox_change(self):
        """Handle focus spinbox changes (arrow clicks)."""
        focus_val = self.manual_focus_var.get()
        if self.preview_running and self.capture:
            self.capture.set(cv2.CAP_PROP_FOCUS, focus_val)

    def on_manual_focus_spinbox_return(self, event=None):
        """Handle focus spinbox Return key or FocusOut events."""
        focus_val = self.manual_focus_var.get()
        if self.preview_running and self.capture:
            self.capture.set(cv2.CAP_PROP_FOCUS, focus_val)

    def on_manual_exposure_change(self, value):
        """Apply exposure to the camera if running."""
        exposure_val = int(float(value))
        if self.preview_running and self.capture:
            # Disable auto exposure to allow manual control
            self.capture.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25) # 0.25 is manual mode in some backends
            self.capture.set(cv2.CAP_PROP_EXPOSURE, exposure_val)

    def on_manual_exposure_spinbox_change(self):
        """Handle exposure spinbox changes (arrow clicks)."""
        exposure_val = self.manual_exposure_var.get()
        if self.preview_running and self.capture:
            self.capture.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
            self.capture.set(cv2.CAP_PROP_EXPOSURE, exposure_val)

    def on_manual_exposure_spinbox_return(self, event=None):
        """Handle exposure spinbox Return key or FocusOut events."""
        exposure_val = self.manual_exposure_var.get()
        if self.preview_running and self.capture:
            self.capture.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
            self.capture.set(cv2.CAP_PROP_EXPOSURE, exposure_val)

    def start_camera_preview(self):
        """Start live camera preview."""
        if self.preview_running:
            return

        # Default to camera 0 since index selector was removed
        camera_index = 0
        # CAP_DSHOW is required on Windows to access UVC camera controls properly
        capture = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
        if not capture.isOpened():
            capture.release()
            messagebox.showerror("Camera Error", f"Could not open camera index {camera_index}")
            return

        capture.set(cv2.CAP_PROP_FRAME_WIDTH, 3840)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 2160)
        capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        # Disable autofocus immediately so we can control it manually
        capture.set(cv2.CAP_PROP_AUTOFOCUS, 0)
        
        # Disable auto exposure and apply current manual exposure value
        capture.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
        capture.set(cv2.CAP_PROP_EXPOSURE, self.manual_exposure_var.get())
        
        # Apply current manual focus value immediately
        capture.set(cv2.CAP_PROP_FOCUS, self.manual_focus_var.get())

        self.capture = capture
        self.preview_running = True
        self.start_camera_btn.configure(state="disabled")
        self.stop_camera_btn.configure(state="normal")
        self.capture_scan_btn.configure(state="normal")
        self.capture_only_btn.configure(state="normal")
        self.phase_var.set("Live preview")
        self.status_var.set(f"Camera {camera_index} connected. Watching live feed.")
        self.schedule_preview_update()

    def stop_camera_preview(self):
        """Stop live camera preview."""
        self.preview_running = False
        if self.preview_job is not None:
            self.root.after_cancel(self.preview_job)
            self.preview_job = None
        if self.capture is not None:
            self.capture.release()
            self.capture = None

        self.start_camera_btn.configure(state="normal")
        self.stop_camera_btn.configure(state="disabled")
        self.capture_scan_btn.configure(state="disabled")
        self.capture_only_btn.configure(state="disabled")
        self.phase_var.set("Idle")
        self.status_var.set("Camera preview stopped")

    def schedule_preview_update(self):
        """Schedule the next preview frame update."""
        self.update_preview_frame()
        if self.preview_running:
            self.preview_job = self.root.after(40, self.schedule_preview_update)

    def update_preview_frame(self):
        """Read and show one live frame."""
        if not self.capture:
            return

        ok, frame = self.capture.read()
        if not ok or frame is None:
            self.status_var.set("Camera frame read failed")
            return

        self.current_frame = frame
        annotated = frame.copy()
        cv2.putText(
            annotated,
            "Live Preview",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
        self.display_cv_image(annotated)

    def select_image(self):
        """Open file dialog to select an image."""
        filetypes = [
            ("Image files", "*.png *.jpg *.jpeg *.bmp *.tiff *.heic *.pdf"),
            ("PNG files", "*.png"),
            ("JPEG files", "*.jpg *.jpeg"),
            ("All files", "*.*"),
        ]

        filename = filedialog.askopenfilename(title="Select Plate Image", filetypes=filetypes)
        if filename:
            self.current_image_path = filename
            self.load_image(filename)
            self.scan_btn.configure(state="normal")
            self.phase_var.set("Image loaded")
            self.status_var.set(f"Loaded image: {os.path.basename(filename)}")

    def load_image(self, path):
        """Load and display the selected image."""
        try:
            img = Image.open(path)
            if img.mode in ("RGBA", "LA", "P"):
                img = img.convert("RGB")
            elif img.mode == "I;16":
                img = img.convert("I").convert("RGB")
            self.current_display_image = img.copy()
            self.display_pil_image(img)
        except Exception as exc:
            messagebox.showerror("Error", f"Failed to load image:\n{str(exc)}")

    def display_pil_image(self, pil_image):
        """Display a PIL image on the preview canvas."""
        display = pil_image.copy()
        display.thumbnail(self.PREVIEW_SIZE, Image.Resampling.LANCZOS)
        self.preview_photo = ImageTk.PhotoImage(display)
        self.canvas.delete("all")
        self.canvas.configure(width=self.preview_photo.width(), height=self.preview_photo.height())
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.preview_photo)

    def display_cv_image(self, frame):
        """Display an OpenCV BGR image on the preview canvas."""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)
        self.display_pil_image(img)

    def set_busy(self, busy):
        """Enable or disable actions while scanning."""
        state = "disabled" if busy else "normal"
        self.select_btn.configure(state=state)
        self.scan_btn.configure(state=state if self.current_image_path else "disabled")
        self.capture_scan_btn.configure(
            state="disabled" if busy or not self.preview_running else "normal"
        )
        self.capture_only_btn.configure(
            state="disabled" if busy or not self.preview_running else "normal"
        )
        self.start_camera_btn.configure(state="disabled" if busy or self.preview_running else "normal")
        self.stop_camera_btn.configure(state="disabled" if busy or not self.preview_running else "normal")

    def capture_only_save(self):
        """Capture full focus sweep and save all frames to photos directory without scanning."""
        if not self.preview_running or self.capture is None:
            messagebox.showwarning("Warning", "Start the camera preview first")
            return

        if self.scan_thread and self.scan_thread.is_alive():
            return

        self.progress_var.set(0)
        self.phase_var.set("Preparing capture sweep")
        self.status_var.set("Capturing focus sweep - saving all frames")
        self.capture_info_var.set("Saving all frames from focus sweep to photos directory...")
        self.set_busy(True)
        self.scan_thread = threading.Thread(target=self.run_capture_only_workflow, daemon=True)
        self.scan_thread.start()

    def run_capture_only_workflow(self):
        """Sweep hardware focus, capture all frames, and save them without scanning."""
        try:
            focus_start = self.focus_start_var.get()
            focus_end = self.focus_end_var.get()
            sweep_count = max(1, int(self.sweep_count_var.get()))
            delay_seconds = max(0.05, self.capture_delay_var.get() / 1000.0)
            exposure_val = self.manual_exposure_var.get()

            # Create photos directory if it doesn't exist
            photos_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "photos")
            os.makedirs(photos_dir, exist_ok=True)

            # Generate base timestamp for this sweep
            base_timestamp = time.strftime("%Y%m%d_%H%M%S")

            # Calculate the focus values to sweep through
            if sweep_count == 1:
                focus_steps = [focus_start]
            else:
                step_size = (focus_end - focus_start) / (sweep_count - 1)
                focus_steps = [int(focus_start + i * step_size) for i in range(sweep_count)]

            saved_files = []

            for index, focus_val in enumerate(focus_steps):
                # 1. Command the lens motor to move to the absolute focus position
                self.capture.set(cv2.CAP_PROP_FOCUS, focus_val)

                # 2. Wait for the physical voice coil motor to move and settle
                time.sleep(delay_seconds)

                # 3. Flush the buffer (read a few frames to ensure we get the newly focused image)
                for _ in range(3):
                    self.capture.read()

                # 4. Capture the actual frame
                ok, frame = self.capture.read()
                if not ok or frame is None:
                    raise RuntimeError("Failed to capture frame from camera")

                self.current_frame = frame.copy()
                sharpness = self.scanner.calculate_focus_score(frame)

                # 5. Save the frame with clear naming
                filename = f"SWEEP_{base_timestamp}_F{index+1:02d}_EXP{exposure_val}_FOC{focus_val}_SHP{sharpness:.0f}.jpg"
                filepath = os.path.join(photos_dir, filename)
                cv2.imwrite(filepath, frame)
                saved_files.append(filename)

                # Update UI progress
                progress = (index + 1) / sweep_count * 100
                self.root.after(0, lambda p=progress, f=filename: self._update_capture_progress(p, f))

                # Show preview with capture info
                annotated = self.scanner.create_preview_overlay(
                    frame,
                    f"Capture {index + 1}/{sweep_count} | Focus: {focus_val} | Sharpness: {sharpness:.1f}",
                    scale=0.70,
                )
                self.root.after(0, lambda img=annotated: self.display_cv_image(img))

            # Final status update
            self.root.after(0, lambda: self._finish_capture_only(saved_files, photos_dir))

        except Exception as exc:
            self.root.after(0, lambda e=str(exc): self._handle_capture_error(e))

    def _update_capture_progress(self, progress, filename):
        """Update progress bar during capture-only workflow."""
        self.progress_var.set(progress)
        self.status_var.set(f"Saved: {filename}")

    def _finish_capture_only(self, saved_files, photos_dir):
        """Complete the capture-only workflow."""
        self.progress_var.set(100)
        self.phase_var.set("Capture complete")
        self.status_var.set(f"Saved {len(saved_files)} images to photos directory")
        self.capture_info_var.set(f"Focus sweep complete. {len(saved_files)} images saved.")
        self.set_busy(False)

    def _handle_capture_error(self, error_msg):
        """Handle errors during capture-only workflow."""
        self.progress_var.set(0)
        self.phase_var.set("Error")
        self.status_var.set(f"Capture failed: {error_msg}")
        self.set_busy(False)
        messagebox.showerror("Capture Error", f"Failed during capture:\n{error_msg}")

    def start_live_scan(self):
        """Capture multiple live frames and scan them in a background thread."""
        if not self.preview_running or self.capture is None:
            messagebox.showwarning("Warning", "Start the camera preview first")
            return

        if self.scan_thread and self.scan_thread.is_alive():
            return

        self.progress_var.set(0)
        self.last_stack_metadata = []
        self.phase_var.set("Preparing capture sweep")
        self.status_var.set("Sweeping hardware focus and collecting frames")
        self.capture_info_var.set(
            "Capturing several frames over time. For each well the best decode from the stack is retained."
        )
        self.set_busy(True)
        self.scan_thread = threading.Thread(target=self.run_live_scan_workflow, daemon=True)
        self.scan_thread.start()

    def run_live_scan_workflow(self):
        """Sweep hardware focus, capture a frame stack, and merge best barcode results."""
        try:
            focus_start = self.focus_start_var.get()
            focus_end = self.focus_end_var.get()
            sweep_count = max(1, int(self.sweep_count_var.get()))
            delay_seconds = max(0.05, self.capture_delay_var.get() / 1000.0)

            apply_correction = self.distortion_var.get()
            k1 = self.k1_var.get()
            k2 = self.k2_var.get()
            # Default preview scale since selector was removed
            preview_scale = 0.70

            # Calculate the focus values to sweep through
            if sweep_count == 1:
                focus_steps = [focus_start]
            else:
                step_size = (focus_end - focus_start) / (sweep_count - 1)
                focus_steps = [int(focus_start + i * step_size) for i in range(sweep_count)]

            # Use streaming mode for large sweeps to avoid memory crashes
            # Streaming processes frames one at a time instead of keeping all in memory
            use_streaming = sweep_count >= 10
            
            if use_streaming:
                self.root.after(
                    0,
                    self.update_progress,
                    5,
                    "Capturing (streaming mode)",
                    f"Large sweep ({sweep_count} frames) - using memory-efficient streaming",
                )
                
                # Generator that yields frames one at a time
                def frame_generator():
                    for index, focus_val in enumerate(focus_steps):
                        # 1. Command the lens motor to move
                        self.capture.set(cv2.CAP_PROP_FOCUS, focus_val)
                        # 2. Wait for motor to settle
                        time.sleep(delay_seconds)
                        # 3. Flush buffer
                        for _ in range(3):
                            self.capture.read()
                        # 4. Capture frame
                        ok, frame = self.capture.read()
                        if ok and frame is not None:
                            yield (index + 1, frame)
                        # Frame goes out of scope, can be garbage collected
                
                # Process using streaming method
                results, stack_metadata, composite = self.scanner.scan_frame_streaming(
                    frame_generator(),
                    sweep_count,
                    apply_distortion_correction=apply_correction,
                    k1=k1,
                    k2=k2,
                    progress_callback=lambda current, total, message: self.root.after(
                        0,
                        self.update_progress,
                        5 + (current / total) * 90,
                        "Analyzing (streaming)",
                        message,
                    ),
                )
                
                # Build metadata for summary
                metadata = [{"index": i+1, "focus_val": fv, "sharpness": 0} 
                           for i, fv in enumerate(focus_steps)]
            else:
                # Original method for small sweeps (keeps all frames in memory)
                frames = []
                metadata = []

                for index, focus_val in enumerate(focus_steps):
                    # 1. Command the lens motor to move to the absolute focus position
                    self.capture.set(cv2.CAP_PROP_FOCUS, focus_val)

                    # 2. Wait for the physical voice coil motor to move and settle
                    time.sleep(delay_seconds)

                    # 3. Flush the buffer (read a few frames to ensure we get the newly focused image)
                    for _ in range(3):
                        self.capture.read()

                    # 4. Capture the actual frame
                    ok, frame = self.capture.read()
                    if not ok or frame is None:
                        raise RuntimeError("Failed to capture frame from camera")

                    self.current_frame = frame.copy()
                    sharpness = self.scanner.calculate_focus_score(frame)
                    frames.append(frame.copy())
                    metadata.append({"index": index + 1, "focus_val": focus_val, "sharpness": sharpness})

                    annotated = self.scanner.create_preview_overlay(
                        frame,
                        f"Capture {index + 1}/{sweep_count} | Focus: {focus_val} | Sharpness: {sharpness:.1f}",
                        scale=preview_scale,
                    )
                    self.root.after(0, self.display_cv_image, annotated)
                    self.root.after(
                        0,
                        self.update_progress,
                        ((index + 1) / sweep_count) * 45,
                        "Capturing focus sweep",
                        f"Captured frame {index + 1} of {sweep_count} (Focus: {focus_val})",
                    )

                # 5. Process the stack
                results, stack_metadata, composite = self.scanner.scan_frame_stack(
                    frames,
                    apply_distortion_correction=apply_correction,
                    k1=k1,
                    k2=k2,
                    progress_callback=lambda current, total, message: self.root.after(
                        0,
                        self.update_progress,
                        45 + (current / total) * 50,
                        "Analyzing frame stack",
                        message,
                    ),
                )

            self.last_stack_metadata = stack_metadata
            detected = sum(1 for value in results.values() if value)

            with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as temp_file:
                composite_path = temp_file.name
            cv2.imwrite(composite_path, composite)

            self.current_image_path = composite_path
            self.current_results = results
            self.root.after(0, self.load_image, composite_path)
            self.root.after(0, self.display_results, results)
            self.root.after(0, lambda: self.export_txt_btn.configure(state="normal"))
            self.root.after(0, lambda: self.export_csv_btn.configure(state="normal"))
            self.root.after(0, lambda: self.scan_btn.configure(state="normal"))
            self.root.after(
                0,
                self.capture_info_var.set,
                self.format_stack_summary(metadata, stack_metadata, detected, use_streaming),
            )
            self.root.after(
                0,
                self.update_progress,
                100,
                "Complete",
                f"Live scan complete: {detected}/96 barcodes found" + (" (streaming mode)" if use_streaming else ""),
            )
        except Exception as exc:
            self.root.after(0, messagebox.showerror, "Error", f"Live scan failed:\n{str(exc)}")
            self.root.after(0, self.update_progress, 0, "Failed", "Live capture workflow failed")
        finally:
            self.root.after(0, self.set_busy, False)

    def update_progress(self, value, phase, status):
        """Update progress widgets."""
        self.progress_var.set(value)
        self.phase_var.set(phase)
        self.status_var.set(status)

    def scan_plate(self):
        """Scan the loaded image from disk."""
        if not self.current_image_path:
            messagebox.showwarning("Warning", "Please select an image first")
            return

        self.phase_var.set("Scanning image")
        self.status_var.set("Processing selected image")
        self.progress_var.set(15)
        self.root.update()

        try:
            k1 = self.k1_var.get()
            k2 = self.k2_var.get()
            use_correction = self.distortion_var.get()

            results = self.scanner.scan_plate(
                self.current_image_path,
                apply_distortion_correction=use_correction,
                k1=k1,
                k2=k2,
            )

            self.current_results = results
            self.display_results(results)
            self.export_txt_btn.configure(state="normal")
            self.export_csv_btn.configure(state="normal")

            detected = sum(1 for value in results.values() if value is not None)
            self.progress_var.set(100)
            self.phase_var.set("Complete")
            self.status_var.set(f"Image scan complete: {detected}/96 barcodes found")
        except Exception as exc:
            messagebox.showerror("Error", f"Scan failed:\n{str(exc)}")
            self.phase_var.set("Failed")
            self.status_var.set("Image scan failed")

    def display_results(self, results):
        """Display scan results in the GUI."""
        detected_count = 0
        for well_id, barcode in results.items():
            cell = self.grid_cells.get(well_id)
            if cell:
                if barcode:
                    cell.configure(bg="lightgreen", text="✓")
                    detected_count += 1
                else:
                    cell.configure(bg="lightcoral", text="")

        self.stats_var.set(f"Detected: {detected_count}/96")
        self.results_text.delete(1.0, tk.END)

        def sort_key(well_id):
            return well_id[0], int(well_id[1:])

        for well_id in sorted(results.keys(), key=sort_key):
            barcode = results[well_id]
            if barcode:
                self.results_text.insert(tk.END, f"{well_id}: {barcode}\n")

    def format_stack_summary(self, capture_metadata, scan_metadata, detected, streaming_mode=False):
        """Create a human-readable summary for the focus sweep."""
        capture_text = ", ".join(
            f"F={item['focus_val']}({item.get('sharpness', 0):.0f})" for item in capture_metadata
        )
        best_frames = sorted(
            scan_metadata,
            key=lambda item: item.get("decoded_count", 0),
            reverse=True,
        )[:3]
        best_text = ", ".join(
            f"frame {item['frame_index']} -> {item['decoded_count']} wells"
            for item in best_frames
        ) or "no successful decodes"
        mode_text = " [STREAMING MODE - Memory efficient]" if streaming_mode else ""
        return (
            f"Focus sweeps: {capture_text}. Best barcode frames: {best_text}. "
            f"Final merged result detected {detected}/96 wells.{mode_text}"
        )

    def export_results(self, format_type):
        """Export results to file."""
        if not self.current_results:
            messagebox.showwarning("Warning", "No results to export")
            return

        if format_type == "txt":
            filetypes = [("Text files", "*.txt"), ("All files", "*.*")]
            default_ext = ".txt"
        else:
            filetypes = [("CSV files", "*.csv"), ("All files", "*.*")]
            default_ext = ".csv"

        filename = filedialog.asksaveasfilename(defaultextension=default_ext, filetypes=filetypes)
        if not filename:
            return

        try:
            def sort_key(well_id):
                return well_id[0], int(well_id[1:])

            sorted_wells = sorted(self.current_results.keys(), key=sort_key)
            with open(filename, "w", encoding="utf-8") as output_file:
                if format_type == "csv":
                    output_file.write("Well,Barcode\n")
                    for well_id in sorted_wells:
                        barcode = self.current_results[well_id] or ""
                        output_file.write(f"{well_id},{barcode}\n")
                else:
                    output_file.write("96-Well Plate Barcode Scan Results\n")
                    output_file.write("=" * 50 + "\n\n")
                    detected = sum(1 for value in self.current_results.values() if value)
                    output_file.write(f"Detected: {detected}/96 barcodes\n\n")
                    for well_id in sorted_wells:
                        barcode = self.current_results[well_id]
                        if barcode:
                            output_file.write(f"{well_id}: {barcode}\n")

            self.status_var.set(f"Exported to: {os.path.basename(filename)}")
            messagebox.showinfo("Success", f"Results exported to:\n{filename}")
        except Exception as exc:
            messagebox.showerror("Error", f"Export failed:\n{str(exc)}")

    def on_close(self):
        """Clean up resources before closing."""
        self.stop_camera_preview()
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()

    try:
        from ctypes import windll

        windll.shcore.SetProcessDpiAwareness(1)
    except Exception:
        pass

    BarcodeScannerGUI(root)
    root.mainloop()