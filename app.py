import os
import glob
import json
import zipfile
import base64
import io
import importlib
import pandas as pd
import numpy as np
import streamlit as st
import streamlit.components.v1 as components
from PIL import Image
import matplotlib.pyplot as plt
import dicom_processor as dp
import create_dataset as cd

importlib.reload(dp)
importlib.reload(cd)

# Set Page Config
st.set_page_config(
    page_title="MedDICOM Studio - Viewer & Dataset Builder",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling (Dark Glassmorphism Theme)
st.markdown("""
<style>
    /* Dark Mode Glassmorphism Theme */
    .stApp {
        background-color: #0d1117;
        color: #c9d1d9;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Header Card */
    .header-card {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.7), rgba(15, 23, 42, 0.9));
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        padding: 24px;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.5);
        backdrop-filter: blur(12px);
        margin-bottom: 24px;
    }
    
    .header-title {
        color: #38bdf8;
        font-size: 28px;
        font-weight: 700;
        margin: 0;
        display: flex;
        align-items: center;
        gap: 12px;
    }
    
    .header-subtitle {
        color: #94a3b8;
        font-size: 14px;
        margin-top: 6px;
    }
    
    /* Metric Cards */
    .metric-box {
        background: rgba(30, 41, 59, 0.5);
        border: 1px solid rgba(56, 189, 248, 0.2);
        border-radius: 12px;
        padding: 16px;
        text-align: center;
    }
    .metric-value {
        font-size: 26px;
        font-weight: 700;
        color: #38bdf8;
    }
    .metric-label {
        font-size: 12px;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    /* Tabs Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 12px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: rgba(30, 41, 59, 0.5);
        border-radius: 8px;
        color: #94a3b8;
        padding: 8px 20px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #0284c7 !important;
        color: white !important;
    }
</style>
""", unsafe_allow_html=True)

# App Header
st.markdown("""
<div class="header-card">
    <div class="header-title">🩺 MedDICOM Studio</div>
    <div class="header-subtitle">Interactive Medical DICOM Viewer & Machine Learning Dataset Generator</div>
</div>
""", unsafe_allow_html=True)

# Workspace Setup
WORK_DIR = os.getcwd()
EXTRACTED_DIR = os.path.join(WORK_DIR, "extracted_dicoms")

# Extract zips automatically if extracted directory doesn't exist yet
if not os.path.exists(EXTRACTED_DIR):
    dp.extract_zips(WORK_DIR, EXTRACTED_DIR)

dicom_files = dp.find_all_dicom_files(EXTRACTED_DIR)
zip_files = [f for f in os.listdir(WORK_DIR) if f.endswith('.zip')]

# Sidebar Controls
with st.sidebar:
    st.title("📁 File Explorer")
    st.info(f"Workspace: `{WORK_DIR}`")
    
    if st.button("🔄 Rescan / Re-extract ZIPs"):
        dp.extract_zips(WORK_DIR, EXTRACTED_DIR)
        dicom_files = dp.find_all_dicom_files(EXTRACTED_DIR)
        st.success("Extracted & re-scanned successfully!")

    st.subheader("Summary Statistics")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-value">{len(zip_files)}</div>
            <div class="metric-label">ZIP Archives</div>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-value">{len(dicom_files)}</div>
            <div class="metric-label">DICOM Files</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.caption("Developed with pydicom, PIL, Streamlit & PyTorch tools.")

# Main Tabs
tab1, tab2, tab3 = st.tabs(["🔬 Interactive DICOM Viewer", "📦 Dataset Builder Wizard", "📊 Dataset Explorer"])

# TAB 1: INTERACTIVE DICOM READER & VIEWER
with tab1:
    st.markdown("### 🔬 Diagnostic HTML5 DICOM Reader & Multi-Slice Viewer")

    # File Drag-and-Drop Uploader
    uploaded_file = st.file_uploader("📥 Upload custom DICOM file (.dcm) or ZIP archive", type=["dcm", "zip"])
    
    if uploaded_file is not None:
        upload_save_path = os.path.join(EXTRACTED_DIR, "user_uploads", uploaded_file.name)
        os.makedirs(os.path.dirname(upload_save_path), exist_ok=True)
        with open(upload_save_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        if uploaded_file.name.endswith(".zip"):
            dp.extract_zips(os.path.dirname(upload_save_path), EXTRACTED_DIR)
            dicom_files = dp.find_all_dicom_files(EXTRACTED_DIR)
            st.success(f"Extracted uploaded archive `{uploaded_file.name}` successfully!")
        else:
            if upload_save_path not in dicom_files:
                dicom_files.insert(0, upload_save_path)
            st.success(f"Loaded uploaded DICOM file `{uploaded_file.name}`!")

    if not dicom_files:
        st.warning("No DICOM files found! Please place ZIP files containing .dcm images into the workspace directory or upload one above.")
    else:
        col_file, col_opts = st.columns([2, 1])
        
        with col_file:
            rel_files = [os.path.relpath(f, EXTRACTED_DIR) for f in dicom_files]
            selected_rel_path = st.selectbox("Select DICOM File from Dataset", rel_files, index=0)
            selected_dcm_path = os.path.join(EXTRACTED_DIR, selected_rel_path)

        meta = dp.parse_dicom_metadata(selected_dcm_path)

        with col_opts:
            window_preset = st.selectbox("Window Preset", list(dp.WINDOW_PRESETS.keys()), index=0)
            colormap_options = [
                "Rainbow Spectrum",
                "Gist Rainbow",
                "Turbo Rainbow",
                "Spectral Multi-Color",
                "HSV Multi-Color",
                "Jet Thermal",
                "Hot Heatmap",
                "Bone",
                "Viridis",
                "Grayscale"
            ]
            colormap_choice = st.selectbox("🌈 Colormap Palette (Multi-Color)", colormap_options, index=0)

        # Slice / Frame Slider
        num_frames = meta.get("number_of_frames", 1)
        frame_idx = 0
        if num_frames > 1:
            frame_idx = st.slider("🎞️ Multi-Frame / Slice Index", 0, num_frames - 1, 0)

        # Get PIL Image with selected multi-color Rainbow Palette
        display_img = dp.dicom_to_pil(
            selected_dcm_path,
            frame_idx=frame_idx,
            preset=window_preset,
            colormap=colormap_choice
        )

        buffered = io.BytesIO()
        display_img.save(buffered, format="PNG")
        img_b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

        # HTML5 DICOM Diagnostic Canvas Component
        pixel_spacing = meta.get("pixel_spacing", ["1.0", "1.0"])
        ps_x = pixel_spacing[0] if isinstance(pixel_spacing, list) and len(pixel_spacing) > 0 else "1.0"
        ps_y = pixel_spacing[1] if isinstance(pixel_spacing, list) and len(pixel_spacing) > 1 else "1.0"

        html5_viewer_code = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <style>
                body {{
                    margin: 0;
                    padding: 0;
                    background-color: #0b0f19;
                    color: #e2e8f0;
                    font-family: 'Segoe UI', Tahoma, sans-serif;
                    user-select: none;
                }}
                .viewer-container {{
                    display: flex;
                    flex-direction: column;
                    align-items: center;
                    background: #111827;
                    border: 1px solid rgba(56, 189, 248, 0.25);
                    border-radius: 12px;
                    padding: 12px;
                    box-shadow: 0 8px 24px rgba(0,0,0,0.6);
                }}
                .toolbar {{
                    display: flex;
                    gap: 8px;
                    background: rgba(30, 41, 59, 0.8);
                    padding: 8px 16px;
                    border-radius: 8px;
                    margin-bottom: 12px;
                    flex-wrap: wrap;
                    justify-content: center;
                }}
                .tool-btn {{
                    background: #1e293b;
                    border: 1px solid #38bdf8;
                    color: #38bdf8;
                    padding: 6px 12px;
                    border-radius: 6px;
                    cursor: pointer;
                    font-weight: 600;
                    font-size: 13px;
                    transition: all 0.2s ease;
                }}
                .tool-btn:hover, .tool-btn.active {{
                    background: #0284c7;
                    color: white;
                }}
                .canvas-wrapper {{
                    position: relative;
                    border: 2px solid #1e293b;
                    border-radius: 8px;
                    overflow: hidden;
                    background: #000;
                    cursor: crosshair;
                }}
                #dicomCanvas {{
                    display: block;
                }}
                .hud-overlay {{
                    position: absolute;
                    top: 10px;
                    left: 10px;
                    background: rgba(15, 23, 42, 0.85);
                    padding: 6px 12px;
                    border-radius: 6px;
                    font-family: monospace;
                    font-size: 12px;
                    color: #38bdf8;
                    pointer-events: none;
                    border: 1px solid rgba(56, 189, 248, 0.3);
                }}
                .measure-hud {{
                    position: absolute;
                    bottom: 10px;
                    left: 10px;
                    background: rgba(15, 23, 42, 0.85);
                    padding: 6px 12px;
                    border-radius: 6px;
                    font-family: monospace;
                    font-size: 12px;
                    color: #4ade80;
                    pointer-events: none;
                    border: 1px solid rgba(74, 222, 128, 0.3);
                }}
            </style>
        </head>
        <body>
            <div class="viewer-container">
                <div class="toolbar">
                    <button class="tool-btn active" id="btnWindow">☀️ Window/Level Drag</button>
                    <button class="tool-btn" id="btnPan">🔍 Pan & Zoom</button>
                    <button class="tool-btn" id="btnMeasure">📏 Measure Distance</button>
                    <button class="tool-btn" id="btnInvert">🔄 Invert</button>
                    <button class="tool-btn" id="btnRotate">↪️ Rotate 90°</button>
                    <button class="tool-btn" id="btnReset">↩️ Reset View</button>
                </div>
                
                <div class="canvas-wrapper" id="wrapper">
                    <canvas id="dicomCanvas"></canvas>
                    <div class="hud-overlay" id="hud">Pos: (0, 0) | Value: - | Modality: {meta.get('modality', 'N/A')}</div>
                    <div class="measure-hud" id="measureHud" style="display:none;">Distance: 0.0 mm</div>
                </div>
            </div>

            <script>
                const img = new Image();
                img.src = "data:image/png;base64,{img_b64}";

                const canvas = document.getElementById('dicomCanvas');
                const ctx = canvas.getContext('2d');
                const hud = document.getElementById('hud');
                const measureHud = document.getElementById('measureHud');

                let mode = 'window'; // window, pan, measure
                let scale = 1.0;
                let panX = 0, panY = 0;
                let isDragging = false;
                let startX = 0, startY = 0;
                let brightness = 0, contrast = 1.0;
                let isInverted = false;
                let rotationAngle = 0;

                // Measurement state
                let measureStart = null, measureEnd = null;
                const pixelSpacingX = parseFloat("{ps_x}") || 1.0;
                const pixelSpacingY = parseFloat("{ps_y}") || 1.0;

                img.onload = () => {{
                    canvas.width = Math.min(img.width, 700);
                    canvas.height = Math.round(canvas.width * (img.height / img.width));
                    resetView();
                    render();
                }};

                function resetView() {{
                    scale = 1.0;
                    panX = 0;
                    panY = 0;
                    brightness = 0;
                    contrast = 1.0;
                    isInverted = false;
                    rotationAngle = 0;
                    measureStart = null;
                    measureEnd = null;
                    measureHud.style.display = 'none';
                    render();
                }}

                function render() {{
                    ctx.save();
                    ctx.clearRect(0, 0, canvas.width, canvas.height);
                    ctx.translate(canvas.width / 2 + panX, canvas.height / 2 + panY);
                    ctx.scale(scale, scale);
                    ctx.rotate(rotationAngle * Math.PI / 180);

                    // Apply Brightness & Contrast
                    let filterStr = `brightness(${{100 + brightness}}%) contrast(${{contrast * 100}}%)`;
                    if (isInverted) filterStr += ` invert(100%)`;
                    ctx.filter = filterStr;

                    ctx.drawImage(img, -canvas.width / 2, -canvas.height / 2, canvas.width, canvas.height);
                    ctx.restore();

                    // Draw Measurement Ruler Line if active
                    if (measureStart && measureEnd) {{
                        ctx.save();
                        ctx.strokeStyle = '#4ade80';
                        ctx.lineWidth = 2;
                        ctx.beginPath();
                        ctx.moveTo(measureStart.x, measureStart.y);
                        ctx.lineTo(measureEnd.x, measureEnd.y);
                        ctx.stroke();

                        // Endpoint handles
                        ctx.fillStyle = '#4ade80';
                        ctx.beginPath();
                        ctx.arc(measureStart.x, measureStart.y, 4, 0, Math.PI * 2);
                        ctx.arc(measureEnd.x, measureEnd.y, 4, 0, Math.PI * 2);
                        ctx.fill();
                        ctx.restore();
                    }}
                }}

                // Tool Modes Switcher
                document.getElementById('btnWindow').onclick = () => setToolMode('window', 'btnWindow');
                document.getElementById('btnPan').onclick = () => setToolMode('pan', 'btnPan');
                document.getElementById('btnMeasure').onclick = () => setToolMode('measure', 'btnMeasure');
                document.getElementById('btnInvert').onclick = () => {{ isInverted = !isInverted; render(); }};
                document.getElementById('btnRotate').onclick = () => {{ rotationAngle = (rotationAngle + 90) % 360; render(); }};
                document.getElementById('btnReset').onclick = resetView;

                function setToolMode(newMode, btnId) {{
                    mode = newMode;
                    document.querySelectorAll('.tool-btn').forEach(b => b.classList.remove('active'));
                    document.getElementById(btnId).classList.add('active');
                }}

                // Mouse Event Handling
                canvas.onmousedown = (e) => {{
                    isDragging = true;
                    const rect = canvas.getBoundingClientRect();
                    startX = e.clientX - rect.left;
                    startY = e.clientY - rect.top;

                    if (mode === 'measure') {{
                        measureStart = {{ x: startX, y: startY }};
                        measureEnd = {{ x: startX, y: startY }};
                        measureHud.style.display = 'block';
                    }}
                }};

                canvas.onmousemove = (e) => {{
                    const rect = canvas.getBoundingClientRect();
                    const currentX = e.clientX - rect.left;
                    const currentY = e.clientY - rect.top;

                    // Update HUD Pos
                    const imgX = Math.round((currentX / canvas.width) * img.width);
                    const imgY = Math.round((currentY / canvas.height) * img.height);
                    hud.innerText = `Pos: (${{imgX}}, ${{imgY}}) | WC/WW Active | Modality: {meta.get('modality', 'N/A')}`;

                    if (!isDragging) return;

                    const dx = currentX - startX;
                    const dy = currentY - startY;

                    if (mode === 'window') {{
                        brightness += dx * 0.2;
                        contrast = Math.max(0.1, contrast - dy * 0.005);
                        startX = currentX;
                        startY = currentY;
                        render();
                    }} else if (mode === 'pan') {{
                        panX += dx;
                        panY += dy;
                        startX = currentX;
                        startY = currentY;
                        render();
                    }} else if (mode === 'measure') {{
                        measureEnd = {{ x: currentX, y: currentY }};
                        
                        // Calculate distance
                        const pxDistX = Math.abs(measureEnd.x - measureStart.x) * (img.width / canvas.width);
                        const pxDistY = Math.abs(measureEnd.y - measureStart.y) * (img.height / canvas.height);
                        const mmDist = Math.sqrt(Math.pow(pxDistX * pixelSpacingX, 2) + Math.pow(pxDistY * pixelSpacingY, 2));
                        
                        measureHud.innerText = `Distance: ${{mmDist.toFixed(2)}} mm (${{Math.sqrt(pxDistX*pxDistX + pxDistY*pxDistY).toFixed(1)}} px)`;
                        render();
                    }}
                }};

                canvas.onmouseup = () => {{ isDragging = false; }};
                canvas.onmouseleave = () => {{ isDragging = false; }};

                // Mouse Wheel Zooming
                canvas.onwheel = (e) => {{
                    e.preventDefault();
                    const zoomFactor = e.deltaY < 0 ? 1.1 : 0.9;
                    scale = Math.min(Math.max(0.5, scale * zoomFactor), 5.0);
                    render();
                }};
            </script>
        </body>
        </html>
        """

        col_canvas, col_meta = st.columns([3, 2])
        
        with col_canvas:
            st.markdown("#### 🖥️ Diagnostic HTML5 DICOM Reader Canvas")
            components.html(html5_viewer_code, height=580)
            st.caption("💡 **Instructions**: Drag mouse on canvas to adjust Window/Level, right-click/scroll to Zoom & Pan, or select 'Measure Distance' to measure anatomical structures in mm!")

        with col_meta:
            st.markdown("#### 📋 Header Metadata Inspector")
            meta_df = pd.DataFrame(list(meta.items()), columns=["Tag / Parameter", "Value"])
            st.dataframe(meta_df, hide_index=True, use_container_width=True, height=500)


# TAB 2: DATASET BUILDER WIZARD
with tab2:
    st.markdown("### 🛠️ Machine Learning Dataset Generator Wizard")
    st.markdown("Convert all DICOM zip files into structured, normalized image datasets ready for PyTorch, TensorFlow, YOLO, or OpenCV.")

    with st.form("dataset_form"):
        col_d1, col_d2 = st.columns(2)
        
        with col_d1:
            dataset_name = st.text_input("Dataset Directory Name", value="dicom_dataset_export")
            export_format = st.selectbox("Export Image Format", ["png", "jpg", "npy", "png16"], help="png: 8-bit, png16: 16-bit raw, npy: numpy array, jpg: compressed RGB")
            window_preset_export = st.selectbox("Windowing Preset", list(dp.WINDOW_PRESETS.keys()), index=0)
            colormap_export = st.selectbox("🌈 Export Colormap Palette", colormap_options, index=0)

        with col_d2:
            st.markdown("**Train / Val / Test Split Ratios**")
            train_r = st.slider("Train Ratio", 0.5, 0.9, 0.8, step=0.05)
            val_r = st.slider("Validation Ratio", 0.0, 0.3, 0.1, step=0.05)
            test_r = round(1.0 - train_r - val_r, 2)
            st.caption(f"Test Ratio (Auto calculated): **{test_r}**")
            
            random_seed = st.number_input("Random Split Seed", value=42, step=1)
            create_zip = st.checkbox("Bundle as Downloadable ZIP Archive", value=True)

        submit_build = st.form_submit_button("🚀 Generate Dataset Now", use_container_width=True)

    if submit_build:
        with st.spinner("Processing DICOM files and generating dataset..."):
            target_out = os.path.join(WORK_DIR, dataset_name)
            out_dir, zip_archive, df_manifest = cd.build_dataset(
                input_dir=WORK_DIR,
                output_dir=target_out,
                img_format=export_format,
                window_preset=window_preset_export,
                colormap=colormap_export,
                train_ratio=train_r,
                val_ratio=val_r,
                test_ratio=test_r,
                random_seed=random_seed,
                create_zip_archive=create_zip
            )
            
            st.success(f"🎉 Dataset created successfully! Total {len(df_manifest)} samples exported to `{out_dir}`")
            
            st.markdown("#### Dataset Split Distribution")
            st.bar_chart(df_manifest["split"].value_counts())

            if zip_archive and os.path.exists(zip_archive):
                with open(zip_archive, "rb") as zf:
                    st.download_button(
                        label="⬇️ Download Dataset ZIP Archive",
                        data=zf.read(),
                        file_name=os.path.basename(zip_archive),
                        mime="application/zip",
                        use_container_width=True
                    )

# TAB 3: DATASET EXPLORER
with tab3:
    st.markdown("### 📊 Existing Dataset Exports")
    
    export_dirs = [d for d in os.listdir(WORK_DIR) if os.path.isdir(d) and (os.path.exists(os.path.join(d, "dataset_index.csv")) or d == "sample_dataset_export")]
    
    if not export_dirs:
        st.info("No generated datasets found yet. Go to the 'Dataset Builder Wizard' tab to generate one!")
    else:
        selected_exp = st.selectbox("Select Generated Dataset", export_dirs)
        exp_path = os.path.join(WORK_DIR, selected_exp)
        
        index_csv = os.path.join(exp_path, "dataset_index.csv")
        if os.path.exists(index_csv):
            df_idx = pd.read_csv(index_csv)
            st.markdown(f"**Total Samples:** {len(df_idx)}")
            
            c_d1, c_d2 = st.columns([3, 1])
            with c_d1:
                st.dataframe(df_idx, use_container_width=True)
            with c_d2:
                st.markdown("#### Class / Split Breakdown")
                st.dataframe(df_idx["split"].value_counts(), use_container_width=True)
                
                # Download CSV
                csv_bytes = df_idx.to_csv(index=False).encode('utf-8')
                st.download_button("⬇️ Download Metadata CSV", data=csv_bytes, file_name=f"{selected_exp}_index.csv", mime="text/csv")
            
            st.markdown("#### 🖼️ Image Grid Sample Preview")
            sample_imgs = df_idx.head(6)
            grid_cols = st.columns(3)
            for idx, (_, row) in enumerate(sample_imgs.iterrows()):
                img_path = row["export_path"]
                if os.path.exists(img_path) and not img_path.endswith('.npy'):
                    with grid_cols[idx % 3]:
                        st.image(img_path, caption=f"{row['sample_id']} ({row['split']})", use_container_width=True)
