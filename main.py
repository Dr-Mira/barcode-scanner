"""
96-Well Plate Barcode Scanner
Decodes DataMatrix/ECC200 barcodes from images of 96-well plates.
Supports bottom-view photos (A1-H12 layout).
"""

import cv2
import gc
import numpy as np
from pylibdmtx.pylibdmtx import decode
from typing import Dict, List, Tuple, Optional
import os
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial

# Get number of available CPU cores for parallel processing
MAX_WORKERS = multiprocessing.cpu_count()

# Recall-first defaults (speed is intentionally de-prioritized)
DEFAULT_ROTATION_STEP_DEG = 7.5
DEFAULT_ROTATION_RANGE_DEG = 45.0
DEFAULT_ROI_SHIFT_FRACTION = 0.08
DEFAULT_ROI_PADDING_FRACTION = 0.16


def _build_rotation_angles(step_deg: float = DEFAULT_ROTATION_STEP_DEG,
                           range_deg: float = DEFAULT_ROTATION_RANGE_DEG) -> List[float]:
    """Build an angle list that always includes cardinal rotations plus fine offsets."""
    cardinals = [0.0, 90.0, 180.0, 270.0]
    if step_deg <= 0 or range_deg <= 0:
        return cardinals

    step_deg = float(step_deg)
    range_deg = min(float(range_deg), 89.0)
    if step_deg >= 90:
        return cardinals

    angles = cardinals.copy()
    offsets = np.arange(-range_deg, range_deg + 1e-6, step_deg)
    for base in cardinals:
        for offset in offsets:
            angle = (base + float(offset)) % 360.0
            if not any(min(abs(angle - existing), 360.0 - abs(angle - existing)) < 1e-3 for existing in angles):
                angles.append(angle)
    return angles


def _rotate_for_decode(img: np.ndarray, angle_deg: float) -> np.ndarray:
    """Rotate image for decode attempts, using fast paths for cardinal angles."""
    angle = float(angle_deg) % 360.0
    rounded = int(round(angle)) % 360

    if abs(angle - rounded) < 1e-3 and rounded in (0, 90, 180, 270):
        if rounded == 0:
            return img
        if rounded == 90:
            return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        if rounded == 180:
            return cv2.rotate(img, cv2.ROTATE_180)
        return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)

    h, w = img.shape[:2]
    center = (w / 2.0, h / 2.0)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    cos = abs(matrix[0, 0])
    sin = abs(matrix[0, 1])

    bound_w = int((h * sin) + (w * cos))
    bound_h = int((h * cos) + (w * sin))

    matrix[0, 2] += (bound_w / 2.0) - center[0]
    matrix[1, 2] += (bound_h / 2.0) - center[1]

    return cv2.warpAffine(
        img,
        matrix,
        (bound_w, bound_h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )


def _generate_roi_views(roi: np.ndarray,
                        shift_fraction: float = DEFAULT_ROI_SHIFT_FRACTION,
                        padding_fraction: float = DEFAULT_ROI_PADDING_FRACTION) -> List[np.ndarray]:
    """Generate ROI views that help with edge/quiet-zone failures."""
    if len(roi.shape) == 3:
        roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    h, w = roi.shape[:2]
    views = [roi]

    padding_px = int(round(min(h, w) * max(0.0, float(padding_fraction))))
    if padding_px > 0:
        views.append(cv2.copyMakeBorder(
            roi, padding_px, padding_px, padding_px, padding_px,
            borderType=cv2.BORDER_CONSTANT,
            value=255,
        ))
        views.append(cv2.copyMakeBorder(
            roi, padding_px, padding_px, padding_px, padding_px,
            borderType=cv2.BORDER_CONSTANT,
            value=0,
        ))

    shift_px = int(round(min(h, w) * max(0.0, float(shift_fraction))))
    if shift_px > 0:
        shifted_source = cv2.copyMakeBorder(
            roi,
            shift_px,
            shift_px,
            shift_px,
            shift_px,
            borderType=cv2.BORDER_REFLECT101,
        )
        for dy in (-shift_px, 0, shift_px):
            for dx in (-shift_px, 0, shift_px):
                if dx == 0 and dy == 0:
                    continue
                y0 = shift_px + dy
                x0 = shift_px + dx
                views.append(shifted_source[y0:y0 + h, x0:x0 + w])

    center_crop_px = int(round(min(h, w) * 0.06))
    if center_crop_px > 0 and h > center_crop_px * 2 + 4 and w > center_crop_px * 2 + 4:
        center_crop = roi[center_crop_px:h - center_crop_px, center_crop_px:w - center_crop_px]
        views.append(cv2.resize(center_crop, (w, h), interpolation=cv2.INTER_CUBIC))

    return views


def _generate_preprocess_variants(roi: np.ndarray) -> List[np.ndarray]:
    """Generate multiple preprocessing candidates for robust DataMatrix decode."""
    variants = [roi]
    variants.append(cv2.bitwise_not(roi))
    variants.append(cv2.equalizeHist(roi))

    clahe_soft = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe_hard = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
    variants.append(clahe_soft.apply(roi))
    variants.append(clahe_hard.apply(roi))

    blurred = cv2.GaussianBlur(roi, (3, 3), 0)
    variants.append(blurred)

    _, otsu = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, otsu_blur = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    variants.append(otsu)
    variants.append(cv2.bitwise_not(otsu))
    variants.append(otsu_blur)

    adaptive_gauss = cv2.adaptiveThreshold(
        roi, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )
    adaptive_mean = cv2.adaptiveThreshold(
        roi, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, 3
    )
    variants.append(adaptive_gauss)
    variants.append(cv2.bitwise_not(adaptive_gauss))
    variants.append(adaptive_mean)
    variants.append(cv2.bitwise_not(adaptive_mean))

    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
    variants.append(cv2.filter2D(roi, -1, kernel))

    morph_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    variants.append(cv2.morphologyEx(otsu, cv2.MORPH_CLOSE, morph_kernel))
    variants.append(cv2.morphologyEx(otsu, cv2.MORPH_OPEN, morph_kernel))

    return variants


def _decode_roi_high_recall(roi: np.ndarray,
                            timeout: int = 200,
                            rotation_step_deg: float = DEFAULT_ROTATION_STEP_DEG,
                            rotation_range_deg: float = DEFAULT_ROTATION_RANGE_DEG,
                            roi_shift_fraction: float = DEFAULT_ROI_SHIFT_FRACTION,
                            roi_padding_fraction: float = DEFAULT_ROI_PADDING_FRACTION) -> Optional[str]:
    """Recall-focused decode pipeline for difficult ROIs."""
    if roi is None or roi.size == 0:
        return None

    angles = _build_rotation_angles(rotation_step_deg, rotation_range_deg)
    scales = [1.0, 1.5, 2.0, 3.0]

    for view in _generate_roi_views(roi, roi_shift_fraction, roi_padding_fraction):
        for variant in _generate_preprocess_variants(view):
            for scale in scales:
                if scale == 1.0:
                    scaled = variant
                else:
                    scaled = cv2.resize(variant, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

                for angle in angles:
                    try:
                        attempt = _rotate_for_decode(scaled, angle)
                        results = decode(attempt, timeout=timeout, max_count=1)
                        if results:
                            return results[0].data.decode('utf-8')
                    except Exception:
                        continue

    return None


def _decode_barcode_worker(roi_data: Tuple[int, int, np.ndarray], timeout: int = 200,
                           rotation_step_deg: float = DEFAULT_ROTATION_STEP_DEG,
                           rotation_range_deg: float = DEFAULT_ROTATION_RANGE_DEG,
                           roi_shift_fraction: float = DEFAULT_ROI_SHIFT_FRACTION,
                           roi_padding_fraction: float = DEFAULT_ROI_PADDING_FRACTION) -> Tuple[int, int, Optional[str]]:
    """
    Worker function for parallel barcode decoding.
    Must be at module level for multiprocessing pickling.

    Args:
        roi_data: Tuple of (row_idx, col_idx, roi_image)
        timeout: Decode timeout in milliseconds

    Returns:
        Tuple of (row_idx, col_idx, decoded_barcode_or_None)
    """
    row_idx, col_idx, roi = roi_data

    barcode = _decode_roi_high_recall(
        roi,
        timeout=timeout,
        rotation_step_deg=rotation_step_deg,
        rotation_range_deg=rotation_range_deg,
        roi_shift_fraction=roi_shift_fraction,
        roi_padding_fraction=roi_padding_fraction,
    )
    return (row_idx, col_idx, barcode)


class BarcodeScanner:
    """Scanner for DataMatrix barcodes in 96-well plate format."""
    
    # 96-well plate dimensions
    ROWS = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
    COLS = list(range(1, 13))
    TOTAL_WELLS = 96
    
    def __init__(self, rows: int = 8, cols: int = 12):
        """
        Initialize scanner.
        
        Args:
            rows: Number of rows (default 8 for A-H)
            cols: Number of columns (default 12 for 1-12)
        """
        self.rows = rows
        self.cols = cols
        self.last_decode_heatmap = np.zeros((self.rows, self.cols), dtype=np.float32)
        self.last_missing_wells: List[str] = []
        
    def read_image(self, image_path: str) -> Optional[np.ndarray]:
        """
        Read image from file path.
        
        Args:
            image_path: Path to image file
            
        Returns:
            Image as numpy array or None if failed
        """
        try:
            # Try OpenCV first
            img = cv2.imread(image_path)
            if img is not None:
                return img
                
            # Try with PIL for HEIC/other formats
            from PIL import Image
            pil_img = Image.open(image_path)
            # Convert to RGB if necessary
            if pil_img.mode != 'RGB':
                pil_img = pil_img.convert('RGB')
            # Convert to numpy array (OpenCV format)
            img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
            return img
        except Exception as e:
            print(f"Error reading image: {e}")
            return None
    
    def preprocess_image(self, img: np.ndarray, apply_clahe: bool = True,
                          clahe_clip_limit: float = 2.0,
                          apply_denoise: bool = True,
                          denoise_strength: int = 10) -> np.ndarray:
        """
        Preprocess image for barcode detection.

        Args:
            img: Input image
            apply_clahe: Whether to apply CLAHE contrast enhancement
            clahe_clip_limit: CLAHE clip limit (higher = more contrast)
            apply_denoise: Whether to apply denoising
            denoise_strength: Denoising strength (higher = more smoothing)

        Returns:
            Preprocessed image
        """
        # Convert to grayscale
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img

        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        if apply_clahe:
            clahe = cv2.createCLAHE(clipLimit=clahe_clip_limit, tileGridSize=(8, 8))
            gray = clahe.apply(gray)

        # Denoise
        if apply_denoise:
            gray = cv2.fastNlMeansDenoising(gray, None, denoise_strength, 7, 21)

        return gray

    def calculate_focus_score(self, img: np.ndarray) -> float:
        """Estimate focus sharpness using Laplacian variance."""
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img
        return float(cv2.Laplacian(gray, cv2.CV_64F).var())

    def create_preview_overlay(self, img: np.ndarray, message: str, scale: float = 0.7) -> np.ndarray:
        """Resize preview image and draw a status overlay."""
        scale = max(0.2, min(scale, 1.0))
        preview = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        overlay = preview.copy()
        cv2.rectangle(overlay, (0, 0), (preview.shape[1], 56), (0, 0, 0), -1)
        preview = cv2.addWeighted(overlay, 0.45, preview, 0.55, 0)
        cv2.putText(
            preview,
            message,
            (14, 36),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
        return preview
    
    def detect_plate_roi(self, img: np.ndarray) -> Tuple[int, int, int, int]:
        """
        Detect the bounding box of the 96-well plate in the image.
        
        Args:
            img: Input image (grayscale or BGR)
            
        Returns:
            Tuple of (x, y, w, h) representing the plate bounding box.
            If detection fails, returns the full image dimensions.
        """
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img.copy()
            
        # Blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (15, 15), 0)
        
        # Threshold to find the dark plate against a lighter background
        # Using Otsu's adaptive thresholding
        thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 51, 10)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return 0, 0, img.shape[1], img.shape[0]
            
        # Find the largest contour that is roughly rectangular
        max_area = 0
        best_rect = None
        
        img_area = img.shape[0] * img.shape[1]
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            # Plate should be reasonably large (e.g., at least 20% of the image)
            if area > img_area * 0.2 and area > max_area:
                x, y, w, h = cv2.boundingRect(cnt)
                # Check aspect ratio (96-well plate is roughly 3:2)
                aspect_ratio = float(w) / h
                if 1.2 < aspect_ratio < 1.8:
                    max_area = area
                    best_rect = (x, y, w, h)
                    
        if best_rect is not None:
            # Add a small padding to ensure we don't cut off edge wells
            x, y, w, h = best_rect
            pad_x = int(w * 0.02)
            pad_y = int(h * 0.02)
            
            x = max(0, x - pad_x)
            y = max(0, y - pad_y)
            w = min(img.shape[1] - x, w + 2 * pad_x)
            h = min(img.shape[0] - y, h + 2 * pad_y)
            
            return x, y, w, h
            
        return 0, 0, img.shape[1], img.shape[0]

    def detect_wells_grid(self, img: np.ndarray, margin: float = 0.02, plate_roi: Optional[Tuple[int, int, int, int]] = None) -> List[Tuple[int, int, int, int]]:
        """
        Detect potential well positions in the image using grid-based approach.

        Args:
            img: Input image
            margin: Margin percentage to avoid edges (default 0.02 = 2%)
            plate_roi: Optional tuple of (x, y, w, h) for the plate bounding box.
                       If None, uses the entire image.

        Returns:
            List of bounding boxes (x, y, w, h) for each well region
        """
        if plate_roi is not None:
            px, py, pw, ph = plate_roi
        else:
            px, py = 0, 0
            ph, pw = img.shape[:2]

        # Calculate grid cell size
        cell_w = pw // self.cols
        cell_h = ph // self.rows

        wells = []
        # Clamp margin to valid range (0 to 0.5)
        margin = max(0.0, min(margin, 0.5))

        for row in range(self.rows):
            for col in range(self.cols):
                # Calculate cell boundaries with margin
                x1 = px + int(col * cell_w + cell_w * margin)
                y1 = py + int(row * cell_h + cell_h * margin)
                x2 = px + int((col + 1) * cell_w - cell_w * margin)
                y2 = py + int((row + 1) * cell_h - cell_h * margin)

                wells.append((x1, y1, x2 - x1, y2 - y1))

        return wells
    
    def correct_lens_distortion(self, img: np.ndarray, k1: float = -0.3, k2: float = 0.1) -> np.ndarray:
        """
        Apply lens distortion correction.
        
        Args:
            img: Input image
            k1: Primary distortion coefficient
            k2: Secondary distortion coefficient
            
        Returns:
            Corrected image
        """
        h, w = img.shape[:2]
        
        # Camera matrix (assume center of image)
        K = np.array([
            [w, 0, w // 2],
            [0, w, h // 2],
            [0, 0, 1]
        ], dtype=np.float32)
        
        # Distortion coefficients
        D = np.array([k1, k2, 0, 0, 0], dtype=np.float32)
        
        # Get optimal new camera matrix
        new_K, roi = cv2.getOptimalNewCameraMatrix(K, D, (w, h), 1, (w, h))
        
        # Undistort
        map1, map2 = cv2.initUndistortRectifyMap(K, D, None, new_K, (w, h), cv2.CV_32FC1)
        corrected = cv2.remap(img, map1, map2, cv2.INTER_LINEAR)
        
        return corrected
    
    def decode_barcode(self, roi: np.ndarray,
                       decode_timeout: int = 200,
                       rotation_step_deg: float = DEFAULT_ROTATION_STEP_DEG,
                       rotation_range_deg: float = DEFAULT_ROTATION_RANGE_DEG,
                       roi_shift_fraction: float = DEFAULT_ROI_SHIFT_FRACTION,
                       roi_padding_fraction: float = DEFAULT_ROI_PADDING_FRACTION) -> Optional[str]:
        """
        Decode DataMatrix barcode from ROI.

        Args:
            roi: Region of interest containing potential barcode

        Returns:
            Decoded barcode string or None
        """
        return _decode_roi_high_recall(
            roi,
            timeout=decode_timeout,
            rotation_step_deg=rotation_step_deg,
            rotation_range_deg=rotation_range_deg,
            roi_shift_fraction=roi_shift_fraction,
            roi_padding_fraction=roi_padding_fraction,
        )
    
    def scan_plate(self, image_path: str, apply_distortion_correction: bool = True,
                   k1: float = -0.3, k2: float = 0.1, max_workers: int = None,
                   well_margin: float = 0.0, decode_timeout: int = 200,
                   apply_clahe: bool = True, clahe_clip_limit: float = 2.0,
                   apply_denoise: bool = True, denoise_strength: int = 10,
                   rotation_step_deg: float = DEFAULT_ROTATION_STEP_DEG,
                   rotation_range_deg: float = DEFAULT_ROTATION_RANGE_DEG,
                   roi_shift_fraction: float = DEFAULT_ROI_SHIFT_FRACTION,
                   roi_padding_fraction: float = DEFAULT_ROI_PADDING_FRACTION) -> Dict[str, Optional[str]]:
        """
        Scan a 96-well plate image for DataMatrix barcodes using parallel processing.

        Args:
            image_path: Path to the image file
            apply_distortion_correction: Whether to apply lens distortion correction
            k1: Primary distortion coefficient
            k2: Secondary distortion coefficient
            max_workers: Number of parallel workers (defaults to all CPU cores)
            well_margin: Margin percentage for well ROI extraction (default 0.0 = no shrink)
            decode_timeout: Timeout for barcode decoding in milliseconds
            apply_clahe: Whether to apply CLAHE contrast enhancement
            clahe_clip_limit: CLAHE clip limit
            apply_denoise: Whether to apply denoising
            denoise_strength: Denoising strength
            rotation_step_deg: Fine rotation step for decode attempts
            rotation_range_deg: Rotation range around each cardinal angle
            roi_shift_fraction: Fractional ROI jitter to catch edge-located barcodes
            roi_padding_fraction: Fractional constant border padding to restore quiet zones

        Returns:
            Dictionary mapping well positions (e.g., 'A1') to barcode values
        """
        # Read image
        img = self.read_image(image_path)
        if img is None:
            raise ValueError(f"Could not read image: {image_path}")

        # Apply distortion correction if requested
        if apply_distortion_correction:
            img = self.correct_lens_distortion(img, k1, k2)

        # Preprocess with configurable parameters
        gray = self.preprocess_image(img, apply_clahe, clahe_clip_limit,
                                      apply_denoise, denoise_strength)

        # Detect plate ROI
        plate_roi = self.detect_plate_roi(gray)

        # Detect well grid with configurable margin
        wells = self.detect_wells_grid(gray, well_margin, plate_roi)
        
        # Prepare data for parallel processing
        roi_data_list = []
        well_ids = []
        for i, (x, y, w, h) in enumerate(wells):
            # Calculate well position (A1, A2, etc.)
            row_idx = i // self.cols
            col_idx = i % self.cols
            
            # For bottom view: left-to-right is A12→A1, B12→B1, etc.
            # So we reverse the column index
            actual_col = self.cols - col_idx
            well_id = f"{self.ROWS[row_idx]}{actual_col}"
            
            # Extract ROI
            roi = gray[y:y+h, x:x+w]
            roi_data_list.append((row_idx, col_idx, roi))
            well_ids.append(well_id)
        
        # Use parallel processing with all available CPU cores
        workers = max_workers if max_workers is not None else MAX_WORKERS
        results = {}
        
        with ProcessPoolExecutor(max_workers=workers) as executor:
            # Submit all tasks with timeout parameter
            future_to_index = {
                executor.submit(
                    partial(
                        _decode_barcode_worker,
                        timeout=decode_timeout,
                        rotation_step_deg=rotation_step_deg,
                        rotation_range_deg=rotation_range_deg,
                        roi_shift_fraction=roi_shift_fraction,
                        roi_padding_fraction=roi_padding_fraction,
                    ),
                    roi_data,
                ): i
                for i, roi_data in enumerate(roi_data_list)
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_index):
                i = future_to_index[future]
                try:
                    row_idx, col_idx, barcode = future.result()
                    results[well_ids[i]] = barcode
                except Exception as e:
                    print(f"Error decoding well {well_ids[i]}: {e}")
                    results[well_ids[i]] = None
        
        self._update_last_decode_heatmap(results)

        # Force garbage collection after parallel processing
        gc.collect()
        return results

    def scan_frame(self, frame: np.ndarray, apply_distortion_correction: bool = False,
                   k1: float = -0.15, k2: float = 0.05, max_workers: int = None,
                   well_margin: float = 0.0, decode_timeout: int = 200,
                   apply_clahe: bool = True, clahe_clip_limit: float = 2.0,
                   apply_denoise: bool = True, denoise_strength: int = 10,
                   rotation_step_deg: float = DEFAULT_ROTATION_STEP_DEG,
                   rotation_range_deg: float = DEFAULT_ROTATION_RANGE_DEG,
                   roi_shift_fraction: float = DEFAULT_ROI_SHIFT_FRACTION,
                   roi_padding_fraction: float = DEFAULT_ROI_PADDING_FRACTION) -> Dict[str, Optional[str]]:
        """Scan a single in-memory frame using parallel processing."""
        working = frame.copy()
        if apply_distortion_correction:
            working = self.correct_lens_distortion(working, k1, k2)

        gray = self.preprocess_image(working, apply_clahe, clahe_clip_limit,
                                      apply_denoise, denoise_strength)
        plate_roi = self.detect_plate_roi(gray)
        wells = self.detect_wells_grid(gray, well_margin, plate_roi)

        # Prepare data for parallel processing
        roi_data_list = []
        well_ids = []
        for i, (x, y, w, h) in enumerate(wells):
            row_idx = i // self.cols
            col_idx = i % self.cols
            actual_col = self.cols - col_idx
            well_id = f"{self.ROWS[row_idx]}{actual_col}"
            roi = gray[y:y + h, x:x + w]
            roi_data_list.append((row_idx, col_idx, roi))
            well_ids.append(well_id)

        # Use parallel processing with all available CPU cores
        workers = max_workers if max_workers is not None else MAX_WORKERS
        results = {}
        
        with ProcessPoolExecutor(max_workers=workers) as executor:
            # Submit all tasks with timeout parameter
            future_to_index = {
                executor.submit(
                    partial(
                        _decode_barcode_worker,
                        timeout=decode_timeout,
                        rotation_step_deg=rotation_step_deg,
                        rotation_range_deg=rotation_range_deg,
                        roi_shift_fraction=roi_shift_fraction,
                        roi_padding_fraction=roi_padding_fraction,
                    ),
                    roi_data,
                ): i
                for i, roi_data in enumerate(roi_data_list)
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_index):
                i = future_to_index[future]
                try:
                    row_idx, col_idx, barcode = future.result()
                    results[well_ids[i]] = barcode
                except Exception as e:
                    print(f"Error decoding well {well_ids[i]}: {e}")
                    results[well_ids[i]] = None

        self._update_last_decode_heatmap(results)

        # Force garbage collection after parallel processing
        gc.collect()
        return results

    def build_focus_composite(self, frames: List[np.ndarray]) -> np.ndarray:
        """Build a simple pixel-wise sharpness composite from multiple frames."""
        if not frames:
            raise ValueError("No frames supplied for composite generation")

        float_frames = [frame.astype(np.float32) for frame in frames]
        gray_frames = [cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) for frame in frames]
        focus_maps = []
        for gray in gray_frames:
            lap = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)
            focus_maps.append(cv2.GaussianBlur(np.abs(lap), (5, 5), 0))

        focus_stack = np.stack(focus_maps, axis=0)
        best_indices = np.argmax(focus_stack, axis=0)
        composite = np.zeros_like(float_frames[0])

        for frame_index, frame in enumerate(float_frames):
            mask = best_indices == frame_index
            composite[mask] = frame[mask]

        return np.clip(composite, 0, 255).astype(np.uint8)

    def scan_frame_stack(self, frames: List[np.ndarray], apply_distortion_correction: bool = False,
                         k1: float = -0.15, k2: float = 0.05, progress_callback=None, max_workers: int = None,
                         well_margin: float = 0.0, decode_timeout: int = 200,
                         apply_clahe: bool = True, clahe_clip_limit: float = 2.0,
                         apply_denoise: bool = True, denoise_strength: int = 10,
                         rotation_step_deg: float = DEFAULT_ROTATION_STEP_DEG,
                         rotation_range_deg: float = DEFAULT_ROTATION_RANGE_DEG,
                         roi_shift_fraction: float = DEFAULT_ROI_SHIFT_FRACTION,
                         roi_padding_fraction: float = DEFAULT_ROI_PADDING_FRACTION):
        """Scan several frames and merge the best per-well results using parallel processing."""
        if not frames:
            raise ValueError("No captured frames available for scanning")

        merged_results = {f"{row}{col}": None for row in self.ROWS for col in range(1, self.cols + 1)}
        metadata = []

        ordered_frames = sorted(
            enumerate(frames, start=1),
            key=lambda item: self.calculate_focus_score(item[1]),
            reverse=True,
        )

        for position, (frame_index, frame) in enumerate(ordered_frames, start=1):
            if progress_callback:
                progress_callback(position, len(ordered_frames), f"Scanning frame {frame_index}/{len(frames)}")

            frame_results = self.scan_frame(
                frame,
                apply_distortion_correction=apply_distortion_correction,
                k1=k1,
                k2=k2,
                max_workers=max_workers,
                well_margin=well_margin,
                decode_timeout=decode_timeout,
                apply_clahe=apply_clahe,
                clahe_clip_limit=clahe_clip_limit,
                apply_denoise=apply_denoise,
                denoise_strength=denoise_strength,
                rotation_step_deg=rotation_step_deg,
                rotation_range_deg=rotation_range_deg,
                roi_shift_fraction=roi_shift_fraction,
                roi_padding_fraction=roi_padding_fraction,
            )
            decoded_count = sum(1 for value in frame_results.values() if value)
            metadata.append({
                "frame_index": frame_index,
                "focus_score": self.calculate_focus_score(frame),
                "decoded_count": decoded_count,
            })

            for well_id, barcode in frame_results.items():
                if barcode and not merged_results[well_id]:
                    merged_results[well_id] = barcode

        composite = self.build_focus_composite(frames)
        composite_results = self.scan_frame(
            composite,
            apply_distortion_correction=apply_distortion_correction,
            k1=k1,
            k2=k2,
            well_margin=well_margin,
            decode_timeout=decode_timeout,
            apply_clahe=apply_clahe,
            clahe_clip_limit=clahe_clip_limit,
            apply_denoise=apply_denoise,
            denoise_strength=denoise_strength,
            rotation_step_deg=rotation_step_deg,
            rotation_range_deg=rotation_range_deg,
            roi_shift_fraction=roi_shift_fraction,
            roi_padding_fraction=roi_padding_fraction,
        )
        for well_id, barcode in composite_results.items():
            if barcode:
                merged_results[well_id] = barcode

        self._update_last_decode_heatmap(merged_results)

        return merged_results, metadata, composite

    def scan_frame_streaming(self, frames_iter, total_frames: int,
                             apply_distortion_correction: bool = False,
                             k1: float = -0.15, k2: float = 0.05,
                             progress_callback=None, best_frames_dir: Optional[str] = None,
                             max_cached_frames: int = 5, max_workers: int = None,
                             well_margin: float = 0.0, decode_timeout: int = 200,
                             apply_clahe: bool = True, clahe_clip_limit: float = 2.0,
                             apply_denoise: bool = True, denoise_strength: int = 10,
                             rotation_step_deg: float = DEFAULT_ROTATION_STEP_DEG,
                             rotation_range_deg: float = DEFAULT_ROTATION_RANGE_DEG,
                             roi_shift_fraction: float = DEFAULT_ROI_SHIFT_FRACTION,
                             roi_padding_fraction: float = DEFAULT_ROI_PADDING_FRACTION):
        """
        Memory-efficient streaming scan that processes frames one at a time.

        Instead of keeping all frames in memory, this:
        1. Processes each frame immediately
        2. Keeps only the best barcode per well with confidence metadata
        3. Optionally saves the best frames to disk instead of memory
        4. Returns a composite of the best regions without full focus stacking

        Args:
            frames_iter: Iterator yielding (frame_index, focus_val, frame) tuples
            total_frames: Total number of frames to process
            apply_distortion_correction: Whether to apply lens correction
            k1, k2: Distortion coefficients
            progress_callback: Optional callback(current, total, message)
            best_frames_dir: If set, saves best frames to disk instead of keeping in memory
            max_cached_frames: Maximum number of frames to keep in memory (default 5)
            well_margin: Margin percentage for well ROI extraction
            decode_timeout: Timeout for barcode decoding in milliseconds
            apply_clahe: Whether to apply CLAHE contrast enhancement
            clahe_clip_limit: CLAHE clip limit
            apply_denoise: Whether to apply denoising
            denoise_strength: Denoising strength

        Returns:
            (merged_results, metadata, best_composite_frame)
        """
        # Track best result per well with confidence score
        # Format: {well_id: (barcode, confidence_score, frame_index)}
        best_results: Dict[str, Tuple[Optional[str], float, int]] = {
            f"{row}{col}": (None, 0.0, -1)
            for row in self.ROWS
            for col in range(1, self.cols + 1)
        }

        metadata = []
        best_overall_frame = None
        best_overall_score = -1.0

        # For composite building - track which frame had best focus per well region
        # Instead of pixel-wise focus stacking, we use well-wise selection
        well_frame_assignments: Dict[str, int] = {}
        frame_cache: Dict[int, np.ndarray] = {} if best_frames_dir is None else None
        cached_frame_scores: Dict[int, float] = {}  # Track scores for cache eviction

        for position, (frame_index, focus_val, frame) in enumerate(frames_iter, start=1):
            # Calculate focus score for this frame
            focus_score = self.calculate_focus_score(frame)
            
            # Track best overall frame for composite
            if focus_score > best_overall_score:
                best_overall_score = focus_score
                if best_frames_dir is None:
                    best_overall_frame = frame.copy()
                else:
                    # Save to disk instead of memory
                    best_overall_frame = frame  # Keep reference temporarily
            
            if progress_callback:
                progress_callback(position, total_frames,
                    f"Scanning frame {frame_index}/{total_frames} (Focus: {focus_val})")
            
            # Process frame immediately
            frame_results = self.scan_frame(
                frame,
                apply_distortion_correction=apply_distortion_correction,
                k1=k1,
                k2=k2,
                max_workers=max_workers,
                well_margin=well_margin,
                decode_timeout=decode_timeout,
                apply_clahe=apply_clahe,
                clahe_clip_limit=clahe_clip_limit,
                apply_denoise=apply_denoise,
                denoise_strength=denoise_strength,
                rotation_step_deg=rotation_step_deg,
                rotation_range_deg=rotation_range_deg,
                roi_shift_fraction=roi_shift_fraction,
                roi_padding_fraction=roi_padding_fraction,
            )

            decoded_count = sum(1 for value in frame_results.values() if value)
            metadata.append({
                "frame_index": frame_index,
                "focus_val": focus_val,
                "focus_score": focus_score,
                "decoded_count": decoded_count,
            })
            
            # Update best results per well with confidence heuristic
            # Confidence = focus_score * (1 if decoded else 0.5) 
            # This prefers sharp frames but also considers decode success
            for well_id, barcode in frame_results.items():
                confidence = focus_score * (1.0 if barcode else 0.3)
                current_best = best_results[well_id]
                
                if barcode and confidence > current_best[1]:
                    best_results[well_id] = (barcode, confidence, frame_index)
                    well_frame_assignments[well_id] = frame_index
                    
                    # Save frame reference if needed for composite (with size limit)
                    if best_frames_dir is None:
                        # If cache is full, evict lowest score frame
                        if len(frame_cache) >= max_cached_frames and frame_index not in frame_cache:
                            # Find frame with lowest score to evict
                            worst_frame_idx = min(cached_frame_scores, key=cached_frame_scores.get)
                            del frame_cache[worst_frame_idx]
                            del cached_frame_scores[worst_frame_idx]
                        # Add/update frame in cache
                        if frame_index not in frame_cache:
                            frame_cache[frame_index] = frame.copy()
                        cached_frame_scores[frame_index] = confidence
            
            # If using disk storage, save frame if it decoded well
            if best_frames_dir and decoded_count > 0:
                frame_path = os.path.join(best_frames_dir, f"frame_{frame_index:04d}_shp{focus_score:.0f}.jpg")
                cv2.imwrite(frame_path, frame)
                if focus_score > best_overall_score:
                    best_overall_frame = frame

            # Explicitly delete frame to free memory (generator will create new one)
            del frame
            # Periodic garbage collection every 10 frames
            if position % 10 == 0:
                gc.collect()

        # Build final merged results
        merged_results = {well_id: data[0] for well_id, data in best_results.items()}
        
        # Create a simple composite using the single best overall frame
        # This is much more memory efficient than pixel-wise focus stacking
        # Quality is nearly as good because we already selected best barcodes from all frames
        composite = best_overall_frame.copy() if best_overall_frame is not None else None
        
        # If we have cached frames, try to fill in missing wells from their best frames
        if frame_cache and composite is not None:
            gray = cv2.cvtColor(composite, cv2.COLOR_BGR2GRAY)
            plate_roi = self.detect_plate_roi(gray)
            wells = self.detect_wells_grid(gray, well_margin, plate_roi)
            
            for i, (x, y, w, h) in enumerate(wells):
                row_idx = i // self.cols
                col_idx = i % self.cols
                actual_col = self.cols - col_idx
                well_id = f"{self.ROWS[row_idx]}{actual_col}"
                
                if well_id in well_frame_assignments:
                    assigned_frame_idx = well_frame_assignments[well_id]
                    if assigned_frame_idx in frame_cache:
                        assigned_frame = frame_cache[assigned_frame_idx]
                        # Copy the well region from best frame
                        composite[y:y+h, x:x+w] = assigned_frame[y:y+h, x:x+w]
        
        # Clear memory
        if frame_cache is not None:
            frame_cache.clear()
        if cached_frame_scores is not None:
            cached_frame_scores.clear()

        self._update_last_decode_heatmap(merged_results)
        
        return merged_results, metadata, composite

    def scan_plate_from_files_streaming(self, image_paths: List[str],
                                        apply_distortion_correction: bool = False,
                                        k1: float = -0.15, k2: float = 0.05,
                                        progress_callback=None, max_workers: int = None,
                                        well_margin: float = 0.0, decode_timeout: int = 200,
                                        apply_clahe: bool = True, clahe_clip_limit: float = 2.0,
                                        apply_denoise: bool = True, denoise_strength: int = 10,
                                        rotation_step_deg: float = DEFAULT_ROTATION_STEP_DEG,
                                        rotation_range_deg: float = DEFAULT_ROTATION_RANGE_DEG,
                                        roi_shift_fraction: float = DEFAULT_ROI_SHIFT_FRACTION,
                                        roi_padding_fraction: float = DEFAULT_ROI_PADDING_FRACTION) -> Tuple[Dict, List, np.ndarray]:
        """
        Scan multiple images from disk in streaming fashion.
        Processes one image at a time to minimize memory usage.

        Args:
            image_paths: List of paths to images
            apply_distortion_correction: Whether to apply lens correction
            k1, k2: Distortion coefficients
            progress_callback: Optional callback(current, total, message)
            well_margin: Margin percentage for well ROI extraction
            decode_timeout: Timeout for barcode decoding in milliseconds
            apply_clahe: Whether to apply CLAHE contrast enhancement
            clahe_clip_limit: CLAHE clip limit
            apply_denoise: Whether to apply denoising
            denoise_strength: Denoising strength

        Returns:
            (merged_results, metadata, composite_image)
        """
        def frame_generator():
            for idx, path in enumerate(image_paths, start=1):
                frame = self.read_image(path)
                if frame is not None:
                    # Use idx as focus_val for file scanning (no hardware focus)
                    yield (idx, idx, frame)
                # Frame goes out of scope and can be garbage collected

        return self.scan_frame_streaming(
            frame_generator(),
            len(image_paths),
            apply_distortion_correction=apply_distortion_correction,
            k1=k1,
            k2=k2,
            progress_callback=progress_callback,
            max_workers=max_workers,
            well_margin=well_margin,
            decode_timeout=decode_timeout,
            apply_clahe=apply_clahe,
            clahe_clip_limit=clahe_clip_limit,
            apply_denoise=apply_denoise,
            denoise_strength=denoise_strength,
            rotation_step_deg=rotation_step_deg,
            rotation_range_deg=rotation_range_deg,
            roi_shift_fraction=roi_shift_fraction,
            roi_padding_fraction=roi_padding_fraction,
        )
    
    def scan_plate_adaptive(self, image_path: str) -> Dict[str, Optional[str]]:
        """
        Scan plate with adaptive threshold to find barcodes first,
        then map them to well positions.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Dictionary mapping well positions to barcode values
        """
        # Read image
        img = self.read_image(image_path)
        if img is None:
            raise ValueError(f"Could not read image: {image_path}")
        
        # Apply distortion correction
        img = self.correct_lens_distortion(img, -0.3, 0.1)
        
        # Preprocess
        gray = self.preprocess_image(img)
        
        # Find all barcodes in the image using adaptive approach
        # Scan at multiple scales
        all_barcodes = []
        scales = [1.0, 1.5, 0.75, 2.0]
        
        for scale in scales:
            if scale != 1.0:
                resized = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
            else:
                resized = gray
            
            try:
                decoded = decode(resized, timeout=200)
                for d in decoded:
                    # Scale coordinates back to original
                    x = int(d.rect.left / scale)
                    y = int(d.rect.top / scale)
                    w = int(d.rect.width / scale)
                    h = int(d.rect.height / scale)
                    data = d.data.decode('utf-8')
                    
                    # Check if this barcode is already found
                    is_new = True
                    for existing in all_barcodes:
                        if (abs(existing[0] - x) < 20 and abs(existing[1] - y) < 20):
                            is_new = False
                            break
                    
                    if is_new:
                        all_barcodes.append((x, y, w, h, data))
            except Exception as e:
                print(f"Error decoding at scale {scale}: {e}")
        
        # Map barcodes to grid positions
        img_h, img_w = gray.shape
        cell_w = img_w / self.cols
        cell_h = img_h / self.rows
        
        results = {}
        # Initialize all positions as None
        for row in self.ROWS:
            for col in range(1, self.cols + 1):
                results[f"{row}{col}"] = None
        
        # Assign barcodes to wells
        for x, y, w, h, data in all_barcodes:
            # Calculate grid position
            col_idx = int(x / cell_w)
            row_idx = int(y / cell_h)
            
            # Clamp to valid range
            col_idx = max(0, min(col_idx, self.cols - 1))
            row_idx = max(0, min(row_idx, self.rows - 1))
            
            # Convert to well ID (reverse col for bottom view)
            actual_col = self.cols - col_idx
            well_id = f"{self.ROWS[row_idx]}{actual_col}"
            
            results[well_id] = data

        self._update_last_decode_heatmap(results)
        
        return results

    def _update_last_decode_heatmap(self, results: Dict[str, Optional[str]]) -> None:
        """Update binary decode heatmap and list of missing wells from latest results."""
        heatmap = np.zeros((self.rows, self.cols), dtype=np.float32)
        missing = []

        for row_idx, row in enumerate(self.ROWS):
            for col in range(1, self.cols + 1):
                well_id = f"{row}{col}"
                col_idx = self.cols - col
                found = 1.0 if results.get(well_id) else 0.0
                heatmap[row_idx, col_idx] = found
                if found < 0.5:
                    missing.append(well_id)

        self.last_decode_heatmap = heatmap
        self.last_missing_wells = missing

    def get_last_decode_heatmap(self) -> np.ndarray:
        """Get a copy of the latest decode heatmap (1.0=decoded, 0.0=missing)."""
        return self.last_decode_heatmap.copy()

    def save_last_decode_heatmap(self, output_path: str, cell_size: int = 64) -> str:
        """Render and save the last decode heatmap as an image."""
        heatmap = self.get_last_decode_heatmap()
        if heatmap.size == 0:
            raise ValueError("No heatmap data available")

        rows, cols = heatmap.shape
        height = rows * cell_size
        width = cols * cell_size
        canvas = np.zeros((height, width, 3), dtype=np.uint8)

        for row_idx in range(rows):
            for col_idx in range(cols):
                score = float(np.clip(heatmap[row_idx, col_idx], 0.0, 1.0))
                color = (0, int(255 * score), int(255 * (1.0 - score)))  # BGR: red->green

                y0 = row_idx * cell_size
                x0 = col_idx * cell_size
                y1 = y0 + cell_size
                x1 = x0 + cell_size

                cv2.rectangle(canvas, (x0, y0), (x1, y1), color, thickness=-1)
                cv2.rectangle(canvas, (x0, y0), (x1, y1), (40, 40, 40), thickness=1)

                label_row = self.ROWS[row_idx]
                label_col = self.cols - col_idx
                label = f"{label_row}{label_col}"
                text_color = (255, 255, 255)
                cv2.putText(
                    canvas,
                    label,
                    (x0 + 6, y0 + cell_size // 2),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    text_color,
                    1,
                    cv2.LINE_AA,
                )

        cv2.imwrite(output_path, canvas)
        return output_path


def format_results(results: Dict[str, Optional[str]]) -> str:
    """
    Format scan results as a readable string.
    
    Args:
        results: Dictionary of well positions to barcodes
        
    Returns:
        Formatted string output
    """
    lines = []
    lines.append("=" * 50)
    lines.append("96-Well Plate Barcode Scan Results")
    lines.append("=" * 50)
    lines.append("")
    
    # Count detected
    detected = sum(1 for v in results.values() if v is not None)
    lines.append(f"Detected: {detected}/{len(results)} barcodes")
    lines.append("")
    
    # Sort by well position
    def sort_key(well_id):
        row = well_id[0]
        col = int(well_id[1:])
        return (row, col)
    
    sorted_wells = sorted(results.keys(), key=sort_key)
    
    # Display grid
    lines.append("Grid View:")
    lines.append("-" * 50)
    
    # Header
    header = "    " + " ".join(f"{c:>2}" for c in range(12, 0, -1))
    lines.append(header)
    
    for row in BarcodeScanner.ROWS:
        row_data = []
        for col in range(12, 0, -1):
            well_id = f"{row}{col}"
            if results.get(well_id):
                row_data.append("XX")  # Has barcode
            else:
                row_data.append("..")  # Empty
        lines.append(f"{row} | " + " ".join(row_data))
    
    lines.append("")
    lines.append("Detected Barcodes:")
    lines.append("-" * 50)
    
    for well_id in sorted_wells:
        barcode = results[well_id]
        if barcode:
            lines.append(f"{well_id}: {barcode}")
    
    lines.append("")
    lines.append("=" * 50)
    
    return "\n".join(lines)


def main():
    """Main entry point for command-line usage."""
    import sys
    
    # Get image path from command line or use default
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        # Look for default image
        default_names = ['rack_image.jpg', 'plate.jpg', 'IMG_7521.jpg']
        for name in default_names:
            if os.path.exists(name):
                image_path = name
                break
            # Check in photos folder
            photos_path = os.path.join('photos', name)
            if os.path.exists(photos_path):
                image_path = photos_path
                break
        else:
            print("Usage: python main.py <image_path>")
            print("Or place an image named 'rack_image.jpg' in the current directory")
            sys.exit(1)
    
    print(f"Scanning: {image_path}")
    print()
    
    # Create scanner and scan
    scanner = BarcodeScanner()
    
    try:
        results = scanner.scan_plate_adaptive(image_path)
        print(format_results(results))
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
