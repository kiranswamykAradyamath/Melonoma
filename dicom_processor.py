import os
import zipfile
import glob
import logging
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import pydicom
from pydicom.pixel_data_handlers.util import apply_modality_lut

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

WINDOW_PRESETS = {
    "Auto": None,
    "Soft Tissue": (40, 400),
    "Bone": (400, 1800),
    "Lung": (-600, 1500),
    "Brain": (40, 80),
    "Abdomen": (50, 350),
    "Full Range": "full"
}

def extract_zips(input_dir, extract_to_dir):
    """Extract all zip archives in input_dir into extract_to_dir."""
    os.makedirs(extract_to_dir, exist_ok=True)
    zip_files = [f for f in os.listdir(input_dir) if f.lower().endswith(".zip")]
    extracted_paths = []

    for zip_name in zip_files:
        zip_path = os.path.join(input_dir, zip_name)
        archive_basename = os.path.splitext(zip_name)[0]
        out_folder = os.path.join(extract_to_dir, archive_basename)
        os.makedirs(out_folder, exist_ok=True)
        try:
            with zipfile.ZipFile(zip_path, 'r') as zf:
                zf.extractall(out_folder)
            extracted_paths.append(out_folder)
            logger.info(f"Extracted {zip_name} to {out_folder}")
        except Exception as e:
            logger.error(f"Failed to extract {zip_name}: {e}")
            
    return extracted_paths

def is_dicom_file(filepath):
    """Check if a file is a DICOM file by checking header signature or pydicom readability."""
    if not os.path.isfile(filepath):
        return False
    if filepath.endswith('.bat') or filepath.endswith('.zip') or filepath.endswith('.py') or filepath.endswith('.csv'):
        return False
    try:
        with open(filepath, 'rb') as f:
            header = f.read(132)
            if len(header) >= 132 and header[128:132] == b'DICM':
                return True
        # Fallback check
        ds = pydicom.dcmread(filepath, stop_before_pixels=True, force=True)
        return hasattr(ds, 'SOPClassUID') or hasattr(ds, 'PatientID') or hasattr(ds, 'Modality')
    except Exception:
        return False

def find_all_dicom_files(root_dir):
    """Find all DICOM files under root_dir recursively."""
    dicom_files = []
    for dirpath, _, filenames in os.walk(root_dir):
        for f in filenames:
            full_path = os.path.join(dirpath, f)
            if is_dicom_file(full_path):
                dicom_files.append(full_path)
    return sorted(dicom_files)

def parse_dicom_metadata(dicom_file_path):
    """Extract metadata dictionary from a DICOM file."""
    try:
        ds = pydicom.dcmread(dicom_file_path, stop_before_pixels=True, force=True)
        
        def get_val(tag_name, default="N/A"):
            val = getattr(ds, tag_name, default)
            if isinstance(val, pydicom.multival.MultiValue):
                return [str(v) for v in val]
            return str(val) if val is not None else default

        num_frames = getattr(ds, "NumberOfFrames", 1)
        try:
            num_frames = int(num_frames)
        except Exception:
            num_frames = 1

        has_pixel_data = hasattr(ds, "PixelData") or "PixelData" in ds

        meta = {
            "file_path": dicom_file_path,
            "file_name": os.path.basename(dicom_file_path),
            "patient_id": get_val("PatientID"),
            "study_uid": get_val("StudyInstanceUID"),
            "series_uid": get_val("SeriesInstanceUID"),
            "sop_uid": get_val("SOPInstanceUID"),
            "modality": get_val("Modality"),
            "rows": getattr(ds, "Rows", "N/A"),
            "columns": getattr(ds, "Columns", "N/A"),
            "number_of_frames": num_frames,
            "has_pixel_data": has_pixel_data,
            "bits_allocated": get_val("BitsAllocated"),
            "bits_stored": get_val("BitsStored"),
            "rescale_intercept": get_val("RescaleIntercept", 0),
            "rescale_slope": get_val("RescaleSlope", 1),
            "window_center": get_val("WindowCenter"),
            "window_width": get_val("WindowWidth"),
            "pixel_spacing": get_val("PixelSpacing"),
            "slice_thickness": get_val("SliceThickness"),
            "study_description": get_val("StudyDescription"),
            "series_description": get_val("SeriesDescription"),
            "photometric_interpretation": get_val("PhotometricInterpretation", "MONOCHROME2")
        }
        return meta
    except Exception as e:
        logger.error(f"Error parsing DICOM metadata for {dicom_file_path}: {e}")
        return {
            "file_path": dicom_file_path,
            "file_name": os.path.basename(dicom_file_path),
            "has_pixel_data": False,
            "error": str(e)
        }

def read_pixel_array(dicom_file_path, frame_idx=0):
    """Read DICOM pixel array, returning 2D numpy array and dataset object."""
    ds = pydicom.dcmread(dicom_file_path, force=True)
    if not hasattr(ds, 'pixel_array'):
        return None, ds
    
    try:
        arr = ds.pixel_array
    except Exception as e:
        logger.warning(f"Could not unpack pixel array directly for {dicom_file_path}: {e}")
        return None, ds

    # Handle multi-frame
    if arr.ndim == 3 and arr.shape[0] > 1 and arr.shape[-1] not in (3, 4):
        if frame_idx < arr.shape[0]:
            arr = arr[frame_idx]
        else:
            arr = arr[0]
    elif arr.ndim == 4:
        if frame_idx < arr.shape[0]:
            arr = arr[frame_idx]
        else:
            arr = arr[0]
            
    # Apply modality LUT (Rescale Intercept and Slope)
    try:
        arr = apply_modality_lut(arr, ds)
    except Exception:
        slope = getattr(ds, 'RescaleSlope', 1)
        intercept = getattr(ds, 'RescaleIntercept', 0)
        arr = arr.astype(np.float32) * float(slope) + float(intercept)

    return arr, ds

def apply_windowing(pixel_array, window_center=None, window_width=None, auto_preset="Auto"):
    """Apply DICOM window center and width contrast adjustment."""
    if pixel_array is None:
        return None

    arr = pixel_array.astype(np.float32)

    # Check preset
    if auto_preset in WINDOW_PRESETS and WINDOW_PRESETS[auto_preset] is not None:
        if WINDOW_PRESETS[auto_preset] != "full":
            window_center, window_width = WINDOW_PRESETS[auto_preset]

    if window_center is None or window_width is None or auto_preset == "Full Range":
        # Min-Max Normalization
        min_val = np.min(arr)
        max_val = np.max(arr)
        if max_val == min_val:
            return np.zeros_like(arr, dtype=np.uint8)
        norm = (arr - min_val) / (max_val - min_val) * 255.0
        return np.clip(norm, 0, 255).astype(np.uint8)

    # Convert window bounds
    wc = float(window_center)
    ww = float(window_width)

    img_min = wc - ww / 2.0
    img_max = wc + ww / 2.0

    clipped = np.clip(arr, img_min, img_max)
    if img_max == img_min:
        norm = np.zeros_like(clipped)
    else:
        norm = (clipped - img_min) / (img_max - img_min) * 255.0

    return np.clip(norm, 0, 255).astype(np.uint8)

def create_info_placeholder_image(text_info="No Pixel Data"):
    """Create a clean placeholder PIL image when DICOM has no pixel data."""
    img = Image.new('RGB', (512, 512), color=(24, 28, 36))
    draw = ImageDraw.Draw(img)
    draw.rectangle([10, 10, 501, 501], outline=(60, 130, 246), width=2)
    draw.text((30, 230), "DICOM Metadata Object", fill=(255, 255, 255))
    draw.text((30, 260), text_info, fill=(160, 174, 192))
    return img

def apply_colormap(pil_image, colormap_name="Rainbow"):
    """Apply vibrant multi-color pseudo-coloring (Rainbow, Turbo, Spectral, Jet, etc.) to a PIL Image."""
    if pil_image is None or colormap_name.lower() == "grayscale":
        return pil_image.convert("RGB") if pil_image else None

    # Convert to grayscale array
    gray_img = pil_image.convert("L")
    arr_norm = np.array(gray_img, dtype=np.float32) / 255.0

    cmap_mapping = {
        "rainbow": plt.cm.rainbow,
        "gist rainbow": plt.cm.gist_rainbow,
        "turbo rainbow": plt.cm.turbo,
        "spectral multi-color": plt.cm.nipy_spectral,
        "hsv multi-color": plt.cm.hsv,
        "jet thermal": plt.cm.jet,
        "hot heatmap": plt.cm.hot,
        "coolwarm": plt.cm.coolwarm,
        "bone": plt.cm.bone,
        "viridis": plt.cm.viridis,
        "plasma": plt.cm.plasma,
        "magma": plt.cm.magma
    }

    key = colormap_name.lower()
    cm = cmap_mapping.get(key, plt.cm.gist_rainbow)
    
    # Map normalized 0-1 values to RGB (0-255)
    colored_arr = (cm(arr_norm) * 255.0).astype(np.uint8)
    return Image.fromarray(colored_arr[..., :3], mode="RGB")

def dicom_to_pil(dicom_file_path, frame_idx=0, window_center=None, window_width=None, preset="Auto", colormap=None):
    """Convert DICOM file slice into a PIL Image with optional colormap."""
    arr, ds = read_pixel_array(dicom_file_path, frame_idx=frame_idx)

    if arr is None:
        modality = getattr(ds, "Modality", "Metadata Object")
        return create_info_placeholder_image(f"Modality: {modality} (No Pixel Data)")

    # If RGB image
    if arr.ndim == 3 and arr.shape[-1] in (3, 4):
        arr_uint8 = np.clip(arr, 0, 255).astype(np.uint8)
        return Image.fromarray(arr_uint8[..., :3], mode="RGB")

    # If MONOCHROME1 (Inverted)
    photo_interp = getattr(ds, "PhotometricInterpretation", "MONOCHROME2")
    
    # Check header windowing if user didn't pass explicit values
    if window_center is None and hasattr(ds, 'WindowCenter'):
        wc = ds.WindowCenter
        window_center = wc[0] if isinstance(wc, pydicom.multival.MultiValue) else wc
    if window_width is None and hasattr(ds, 'WindowWidth'):
        ww = ds.WindowWidth
        window_width = ww[0] if isinstance(ww, pydicom.multival.MultiValue) else ww

    norm_arr = apply_windowing(arr, window_center=window_center, window_width=window_width, auto_preset=preset)

    if str(photo_interp).upper() == "MONOCHROME1":
        norm_arr = 255 - norm_arr

    img = Image.fromarray(norm_arr, mode="L")
    
    if colormap and colormap.lower() != "grayscale":
        img = apply_colormap(img, colormap_name=colormap)
        
    return img


def get_dicom_summary(dicom_files):
    """Build pandas DataFrame containing metadata for all provided DICOM files."""
    records = []
    for f in dicom_files:
        meta = parse_dicom_metadata(f)
        records.append(meta)
    return pd.DataFrame(records)
