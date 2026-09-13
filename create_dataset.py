import os
import argparse
import random
import shutil
import json
import zipfile
import numpy as np
import pandas as pd
from PIL import Image
import dicom_processor as dp

def build_dataset(
    input_dir=".",
    output_dir="./dataset_export",
    img_format="png",
    window_preset="Auto",
    colormap="Rainbow Spectrum",
    train_ratio=0.8,
    val_ratio=0.1,
    test_ratio=0.1,
    random_seed=42,
    create_zip_archive=True
):
    """
    Build structured machine learning dataset from DICOM files.
    """
    random.seed(random_seed)
    np.random.seed(random_seed)

    print(f"Scanning for ZIP archives and DICOM files in {input_dir}...")
    
    # 1. Extract ZIP files if any
    extracted_dir = os.path.join(input_dir, "extracted_dicoms")
    if any(f.endswith('.zip') for f in os.listdir(input_dir)):
        dp.extract_zips(input_dir, extracted_dir)
        search_path = extracted_dir
    else:
        search_path = input_dir

    # 2. Locate all DICOM files
    dicom_files = dp.find_all_dicom_files(search_path)
    if not dicom_files:
        raise FileNotFoundError(f"No DICOM files found in {input_dir} or {extracted_dir}")

    print(f"Found {len(dicom_files)} DICOM files.")

    # 3. Create Dataset Output Structure
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)

    train_img_dir = os.path.join(output_dir, "train", "images")
    val_img_dir = os.path.join(output_dir, "val", "images")
    test_img_dir = os.path.join(output_dir, "test", "images")

    os.makedirs(train_img_dir, exist_ok=True)
    os.makedirs(val_img_dir, exist_ok=True)
    os.makedirs(test_img_dir, exist_ok=True)

    # 4. Shuffle & Split
    shuffled_files = list(dicom_files)
    random.shuffle(shuffled_files)

    n_total = len(shuffled_files)
    n_train = max(1, int(n_total * train_ratio)) if n_total > 1 else 1
    n_val = int(n_total * val_ratio)
    n_test = n_total - n_train - n_val

    # Assign split tags
    splits = ["train"] * n_train + ["val"] * n_val + ["test"] * n_test
    if len(splits) < n_total:
        splits += ["test"] * (n_total - len(splits))
    splits = splits[:n_total]

    file_manifest = []

    for i, (dcm_path, split) in enumerate(zip(shuffled_files, splits)):
        meta = dp.parse_dicom_metadata(dcm_path)
        base_name = os.path.splitext(meta["file_name"])[0]
        sample_id = f"sample_{i+1:04d}_{base_name}"
        
        target_dir = os.path.join(output_dir, split, "images")
        
        ext = img_format.lower()
        if ext == "png16":
            ext_str = "png"
        else:
            ext_str = ext

        out_img_filename = f"{sample_id}.{ext_str}"
        out_img_path = os.path.join(target_dir, out_img_filename)

        # Process and Save Image
        try:
            if ext == "npy":
                arr, _ = dp.read_pixel_array(dcm_path)
                if arr is None:
                    arr = np.zeros((512, 512), dtype=np.float32)
                np.save(out_img_path, arr)
            elif ext == "png16":
                arr, _ = dp.read_pixel_array(dcm_path)
                if arr is None:
                    pil_img = dp.dicom_to_pil(dcm_path, colormap=colormap)
                    arr = np.array(pil_img, dtype=np.uint16) * 256
                else:
                    min_val, max_val = np.min(arr), np.max(arr)
                    if max_val > min_val:
                        arr = ((arr - min_val) / (max_val - min_val) * 65535.0).astype(np.uint16)
                    else:
                        arr = np.zeros_like(arr, dtype=np.uint16)
                Image.fromarray(arr).save(out_img_path)
            else:
                # PNG 8-bit or JPEG
                pil_img = dp.dicom_to_pil(dcm_path, preset=window_preset, colormap=colormap)
                if ext == "jpg" or ext == "jpeg":
                    pil_img = pil_img.convert("RGB")
                    pil_img.save(out_img_path, quality=95)
                else:
                    pil_img.save(out_img_path)
            
            meta["sample_id"] = sample_id
            meta["split"] = split
            meta["export_path"] = out_img_path
            meta["export_filename"] = out_img_filename
            file_manifest.append(meta)
            
            print(f"[{i+1}/{n_total}] Processed {meta['file_name']} -> {split}/{out_img_filename}")
        except Exception as e:
            print(f"Error processing {dcm_path}: {e}")

    # 5. Export Master Metadata Index & Split Dataframes
    df_manifest = pd.DataFrame(file_manifest)
    df_manifest.to_csv(os.path.join(output_dir, "dataset_index.csv"), index=False)

    for s in ["train", "val", "test"]:
        df_s = df_manifest[df_manifest["split"] == s]
        df_s.to_csv(os.path.join(output_dir, s, "metadata.csv"), index=False)

    # 6. Export Dataset Summary JSON
    summary = {
        "dataset_name": "DICOM_Medical_Dataset",
        "total_samples": len(file_manifest),
        "split_counts": {
            "train": len(df_manifest[df_manifest["split"] == "train"]),
            "val": len(df_manifest[df_manifest["split"] == "val"]),
            "test": len(df_manifest[df_manifest["split"] == "test"])
        },
        "modalities": df_manifest["modality"].value_counts().to_dict(),
        "export_format": img_format,
        "window_preset": window_preset,
        "random_seed": random_seed
    }

    with open(os.path.join(output_dir, "dataset_summary.json"), "w") as f:
        json.dump(summary, f, indent=4)

    # 7. Create ZIP Bundle if requested
    zip_path = None
    if create_zip_archive:
        zip_path = f"{output_dir}.zip"
        print(f"Creating ZIP archive bundle at {zip_path}...")
        shutil.make_archive(output_dir, 'zip', output_dir)
        print(f"Dataset ZIP archive created successfully!")

    print(f"\nSuccessfully built dataset with {len(file_manifest)} samples in '{output_dir}'.")
    return output_dir, zip_path, df_manifest

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create ML Dataset from DICOM zip files.")
    parser.add_argument("--input", default=".", help="Input directory containing zip files or dicom files")
    parser.add_argument("--output", default="./dataset_export", help="Target output dataset directory")
    parser.add_argument("--format", default="png", choices=["png", "jpg", "npy", "png16"], help="Image format")
    parser.add_argument("--preset", default="Auto", help="DICOM windowing preset")
    parser.add_argument("--train_ratio", type=float, default=0.8, help="Train split ratio")
    parser.add_argument("--val_ratio", type=float, default=0.1, help="Val split ratio")
    parser.add_argument("--test_ratio", type=float, default=0.1, help="Test split ratio")

    args = parser.parse_args()
    build_dataset(
        input_dir=args.input,
        output_dir=args.output,
        img_format=args.format,
        window_preset=args.preset,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio
    )
