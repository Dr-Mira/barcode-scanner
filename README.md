# 96-Well Rack Barcode Scanner

A Python application that decodes Data Matrix barcodes from a 96-well Eppendorf rack image (A1-H12).

## Features

- **🖼️ Visual GUI**: User-friendly interface with image previews
- **👁️ Before/After Preview**: See the original and corrected images side-by-side
- **📊 Visual Detection Grid**: Color-coded grid showing detected (green) vs missing (red) barcodes
- **📐 Bottom-View Support**: Properly handles photos taken from below the rack (A12-A1 → H12-H1 layout)
- **🔧 Lens Distortion Correction**: Corrects barrel/pincushion distortion without checkerboard calibration
- **📐 Perspective Transform**: Optional top-down view correction for angled shots
- **📋 Export Results**: Save scan results to CSV/TXT files

## Installation

1. Install Python 3.7 or higher

2. Install dependencies:
```bash
pip install -r requirements.txt
```

**Note for Windows users**: You may need to install the libdmtx library:
- Download from: https://github.com/dmtx/libdmtx/releases
- Or use: `conda install -c conda-forge libdmtx`

## Usage

### GUI Mode (Recommended)
Launch the graphical interface for visual feedback:
```bash
python gui.py
```

**GUI Features:**
- **Image Preview Panel**: Shows original and distortion-corrected images
- **Visual Grid**: 96-well grid with color-coded cells:
  - 🟢 Green = Barcode detected
  - 🔴 Red = Barcode missing
- **Bottom-View Layout**: Grid arranged for photos taken from below the rack (left-to-right: A12→A1, B12→B1, etc.)
- **Real-time Adjustments**: Modify distortion parameters and rescan
- **Export**: Save results to CSV format

### Command Line Mode
For batch processing or automation:
```bash
python main.py rack_image.jpg
```

### With Default Image
If no image path is provided, the script looks for `rack_image.jpg` in the current directory:
```bash
python main.py
```

## Configuration

### Lens Distortion Parameters

The script uses estimated distortion coefficients. You can adjust them in the `main()` function:

```python
distortion_k1 = -0.3  # Negative = barrel distortion correction
distortion_k2 = 0.1   # Secondary correction
```

**Common values:**
- `k1 = -0.3, k2 = 0.1`: Mild barrel distortion correction (default)
- `k1 = -0.5, k2 = 0.2`: Stronger barrel distortion correction
- `k1 = 0.3, k2 = -0.1`: Pincushion distortion correction

### Perspective Correction

If your camera is at an angle, provide the rack corner coordinates:

```python
rack_corners = {
    'top_left': (100, 50),
    'top_right': (1900, 50),
    'bottom_left': (100, 1150),
    'bottom_right': (1900, 1150)
}
```

To find these coordinates:
1. Open your image in an image editor (Paint, Photoshop, etc.)
2. Hover over the center of each corner vial hole
3. Note the X,Y pixel coordinates
4. Update the `rack_corners` dictionary

## Output

The script outputs:
1. **Grid View**: A formatted table showing all 96 positions
2. **Detailed List**: A sorted list of all detected barcodes
3. **Statistics**: Total barcodes detected out of 96

Example output:
```
============================================================
RACK SCAN RESULTS
============================================================

          1     2     3     4     5     6     7     8     9    10    11    12
--------------------------------------------------------------------------------
  A | ABC123  DEF456  GHI789  JKL012  MNO345  PQR678  STU901  VWX234  YZA567  BCD890  EFG123  HIJ456
  B | KLM789  NOP012  QRS345  TUV678  WXY901  ZAB234  CDE567  FGH890  IJK123  LMN456  OPQ789  RST012
  ...

============================================================
Total barcodes detected: 96 / 96
============================================================
```

## Troubleshooting

### No barcodes detected
- Check image quality and lighting
- Ensure barcodes are clearly visible
- Try adjusting distortion coefficients
- Provide rack_corners for perspective correction

### Wrong position mapping
- Verify the rack is centered in the image
- Adjust rack_corners if using perspective correction
- Ensure the rack is not rotated

### Distortion issues
- Try different k1/k2 values
- For wide-angle lenses: use stronger correction (k1 = -0.5)
- For telephoto lenses: use milder correction (k1 = -0.1)

## How It Works

1. **Load Image**: Reads the JPG file
2. **Lens Correction**: Applies radial distortion correction using estimated camera parameters
3. **Perspective Transform**: (Optional) Warps the image to a top-down view
4. **Preprocessing**: Converts to grayscale and enhances contrast
5. **Barcode Detection**: Uses pylibdmtx to decode Data Matrix codes
6. **Grid Mapping**: Calculates which grid cell each barcode belongs to
7. **Output**: Displays formatted results

## License

Free to use and modify.
