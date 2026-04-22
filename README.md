# 96-Well Rack Barcode Scanner

A Python application that decodes Data Matrix barcodes from a 96-well Eppendorf rack image (A1-H12).

- **DataMatrix engine**: Uses `pylibdmtx`/libdmtx to decode **Data Matrix ECC200** symbols.
- **Sweep + stitch strategy**: For multi-frame scans, the scanner sweeps across frames and keeps the best per-well decode, then builds a stitched composite from best regions.
- **Clever zooming**: Each ROI is decoded at multiple scales (`1.0`, `1.5`, `2.0`, `3.0`) to recover small or blurry codes.
- **Geometric correction math**: Applies radial lens correction with OpenCV camera model (`k1`, `k2`) and optional perspective normalization.
- **Robust decode pass**: Tries ROI shifts/padding, multiple preprocess variants (CLAHE, Otsu/adaptive thresholding, sharpen/morph ops), and rotation sweeps around cardinal angles to maximize recall.

## Configuration

### Lens Distortion Parameters

Adjust distortion coefficients in the `main()` function:

```python
distortion_k1 = -0.3  # Negative = barrel distortion correction
distortion_k2 = 0.1   # Secondary correction
```

Common values:
- `k1 = -0.3, k2 = 0.1`: Mild barrel distortion correction (default)
- `k1 = -0.5, k2 = 0.2`: Stronger barrel distortion correction
- `k1 = 0.3, k2 = -0.1`: Pincushion distortion correction

### Perspective Correction

If your camera is at an angle, provide rack corner coordinates:

```python
rack_corners = {
    'top_left': (100, 50),
    'top_right': (1900, 50),
    'bottom_left': (100, 1150),
    'bottom_right': (1900, 1150)
}
```

To find these coordinates:
1. Open your image in an image editor
2. Hover over the center of each corner vial hole
3. Note the X,Y pixel coordinates
4. Update the `rack_corners` dictionary

## Output

The scanner outputs:
1. Grid view for all 96 positions
2. Detailed list of detected barcodes
3. Detection statistics (detected / 96)

Example:
```
============================================================
RACK SCAN RESULTS
============================================================

      1     2     3  ...  12
--------------------------------
A | CODE1 CODE2 CODE3 ... CODE12
B | ...

============================================================
Total barcodes detected: 96 / 96
============================================================
```
