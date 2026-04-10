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
    
    def preprocess_image(self, img: np.ndarray) -> np.ndarray:
        """
        Preprocess image for barcode detection.
        
        Args:
            img: Input image
            
        Returns:
            Preprocessed image
        """
        # Convert to grayscale
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img
            
        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)
        
        # Denoise
        gray = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
        
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
    
    def detect_wells_grid(self, img: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Detect potential well positions in the image using grid-based approach.
        
        Args:
            img: Input image
            
        Returns:
            List of bounding boxes (x, y, w, h) for each well region
        """
        h, w = img.shape[:2]
        
        # Calculate grid cell size
        cell_w = w // self.cols
        cell_h = h // self.rows
        
        wells = []
        margin = 0.1  # 10% margin to avoid edges
        
        for row in range(self.rows):
            for col in range(self.cols):
                # Calculate cell boundaries with margin
                x1 = int(col * cell_w + cell_w * margin)
                y1 = int(row * cell_h + cell_h * margin)
                x2 = int((col + 1) * cell_w - cell_w * margin)
                y2 = int((row + 1) * cell_h - cell_h * margin)
                
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
    
    def decode_barcode(self, roi: np.ndarray) -> Optional[str]:
        """
        Decode DataMatrix barcode from ROI.

        Args:
            roi: Region of interest containing potential barcode

        Returns:
            Decoded barcode string or None
        """
        # Try different preprocessing approaches (generate on-demand to save memory)
        attempt_generators = [
            lambda: roi,  # Original
            lambda: cv2.resize(roi, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC),  # Upscaled
            lambda: cv2.adaptiveThreshold(roi, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2),  # Thresholded
            lambda: cv2.bitwise_not(roi),  # Inverted
        ]

        for i, gen in enumerate(attempt_generators):
            attempt = None
            try:
                attempt = gen()
                results = decode(attempt, timeout=100)
                if results:
                    # Return first successful decode
                    return results[0].data.decode('utf-8')
            except Exception:
                continue
            finally:
                # Clean up intermediate arrays (except original roi)
                if attempt is not None and i > 0:
                    del attempt

        # Force garbage collection to free memory
        gc.collect()
        return None
    
    def scan_plate(self, image_path: str, apply_distortion_correction: bool = True,
                   k1: float = -0.3, k2: float = 0.1) -> Dict[str, Optional[str]]:
        """
        Scan a 96-well plate image for DataMatrix barcodes.
        
        Args:
            image_path: Path to the image file
            apply_distortion_correction: Whether to apply lens distortion correction
            k1: Primary distortion coefficient
            k2: Secondary distortion coefficient
            
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
        
        # Preprocess
        gray = self.preprocess_image(img)
        
        # Detect well grid
        wells = self.detect_wells_grid(gray)
        
        # Scan each well
        results = {}
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
            
            # Try to decode barcode
            barcode = self.decode_barcode(roi)
            results[well_id] = barcode
        
        return results

    def scan_frame(self, frame: np.ndarray, apply_distortion_correction: bool = False,
                   k1: float = -0.15, k2: float = 0.05) -> Dict[str, Optional[str]]:
        """Scan a single in-memory frame."""
        working = frame.copy()
        if apply_distortion_correction:
            working = self.correct_lens_distortion(working, k1, k2)

        gray = self.preprocess_image(working)
        wells = self.detect_wells_grid(gray)

        results = {}
        for i, (x, y, w, h) in enumerate(wells):
            row_idx = i // self.cols
            col_idx = i % self.cols
            actual_col = self.cols - col_idx
            well_id = f"{self.ROWS[row_idx]}{actual_col}"
            roi = gray[y:y + h, x:x + w]
            results[well_id] = self.decode_barcode(roi)
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
                         k1: float = -0.15, k2: float = 0.05, progress_callback=None):
        """Scan several frames and merge the best per-well results."""
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
        )
        for well_id, barcode in composite_results.items():
            if barcode:
                merged_results[well_id] = barcode

        return merged_results, metadata, composite

    def scan_frame_streaming(self, frames_iter, total_frames: int,
                             apply_distortion_correction: bool = False,
                             k1: float = -0.15, k2: float = 0.05,
                             progress_callback=None, best_frames_dir: Optional[str] = None,
                             max_cached_frames: int = 5):
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
            wells = self.detect_wells_grid(gray)
            
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
        
        return merged_results, metadata, composite

    def scan_plate_from_files_streaming(self, image_paths: List[str],
                                        apply_distortion_correction: bool = False,
                                        k1: float = -0.15, k2: float = 0.05,
                                        progress_callback=None) -> Tuple[Dict, List, np.ndarray]:
        """
        Scan multiple images from disk in streaming fashion.
        Processes one image at a time to minimize memory usage.
        
        Args:
            image_paths: List of paths to images
            apply_distortion_correction: Whether to apply lens correction
            k1, k2: Distortion coefficients  
            progress_callback: Optional callback(current, total, message)
            
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
            progress_callback=progress_callback
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
        
        return results


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
