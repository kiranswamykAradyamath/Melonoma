# 🩺 MedDICOM Studio - Viewer & ML Dataset Builder

An interactive medical DICOM viewer, diagnostic HTML5 canvas reader, and machine learning dataset generator built with Python, PyDicom, Streamlit, and OpenCV.

## Features

- **ZIP & DICOM Extraction**: Recursively extracts `.zip` archives containing DICOM (`.dcm`) image files.
- **Diagnostic HTML5 Reader Canvas**:
  - Interactive Window Center & Window Width contrast adjustments (WC/WW drag).
  - Metric measurement ruler tool (`mm` calibration based on DICOM `PixelSpacing`).
  - Zoom (0.5x to 5.0x), Pan, Invert, and 90° Rotation.
  - Real-time HUD displaying pixel coordinates, active windowing, and modality details.
- **Rainbow & Multi-Color Colormaps**:
  - Apply Rainbow Spectrum, Turbo Rainbow, Gist Rainbow, Spectral Multi-Color, HSV, Jet Thermal, Hot Heatmap, and Viridis false-color maps.
- **Machine Learning Dataset Generator**:
  - Convert DICOMs into structured ML datasets (`train/`, `val/`, `test/`).
  - Supports `PNG` (8-bit), `PNG16` (16-bit raw), `JPEG` (RGB), and `NPY` (NumPy arrays).
  - Outputs master index CSV (`dataset_index.csv`), metadata JSON specs, and downloadable ZIP dataset archives.

## Getting Started

### Installation

```bash
git clone <your-repository-url>
cd melo_data
pip install -r requirements.txt
```

### Launch Interactive Web App

```bash
streamlit run app.py
```
Open [http://localhost:8501](http://localhost:8501) in your browser.

### Batch Dataset Generator (CLI)

```bash
python create_dataset.py --input . --output ./dicom_dataset_export --format png --preset Auto --train_ratio 0.8 --val_ratio 0.1 --test_ratio 0.1
```
