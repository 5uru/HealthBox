import streamlit as st
from PIL import Image
import pandas as pd
import copy

# Local imports — ensure these modules are in your PYTHONPATH
from healthbox.cell_detection import make_prediction
from healthbox.wbc_classification import classify_image
from utils import crop_cells

# --- Page Configuration ---
st.set_page_config(
        page_title="Blood Cell Detection",
        page_icon="🩸",
        layout="wide",
        initial_sidebar_state="expanded"
)

# --- Initialize Session State ---
session_keys = ['detection_result', 'detections_list', 'uploaded_image', 'wbc_classifications']
for key in session_keys:
    if key not in st.session_state:
        st.session_state[key] = None

# --- Sidebar ---
st.sidebar.header("⚙️ Model Settings")
confidence_threshold = st.sidebar.slider(
        "Confidence Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.25,
        step=0.05,
        help="Lower values detect more cells (but may include false positives)."
)

# --- Main UI ---
st.title("🩸 Blood Cell Detection & Classification")
st.markdown(
        "Upload a blood smear image to detect RBCs, WBCs, and Platelets using YOLOv8, "
        "then classify detected WBCs into subtypes."
)

# --- File Uploader & Inference ---
uploaded_file = st.file_uploader("Choose an image...", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None and st.button("🔬 Detect & Classify Cells"):
    try:
        # Ensure image is in RGB (handles RGBA, grayscale, etc.)
        image = Image.open(uploaded_file).convert("RGB")
        st.session_state.uploaded_image = image

        # Reset cached results
        st.session_state.detection_result = None
        st.session_state.detections_list = None
        st.session_state.wbc_classifications = None

        with st.spinner("🔍 Detecting blood cells..."):
            result, detections = make_prediction(image, confidence_threshold)
            st.session_state.detection_result = result
            st.session_state.detections_list = detections

    except Exception as e:
        st.error(f"⚠️ Error during detection: {str(e)}")
        st.session_state.detection_result = None

# --- Results Display ---
if st.session_state.detection_result is not None:
    col1, col2 = st.columns([2, 1])  # Wider for image, narrower for stats

    # === Right Column: Controls & Stats ===
    with col2:
        st.header("📊 Results & Controls")

        class_to_plot = st.selectbox(
                "🔍 Highlight cell type:",
                ("All", "RBC", "WBC", "Platelet"),
                index=0,
                help="Filter bounding boxes on the image."
        )

        # Count detections
        detections = st.session_state.detections_list
        total = len(detections)
        wbc_count = sum(1 for d in detections if d["Class"] == "WBC")
        rbc_count = sum(1 for d in detections if d["Class"] == "RBC")
        platelet_count = sum(1 for d in detections if d["Class"] == "Platelet")

        st.markdown(f"**Total Cells:** {total}")
        st.markdown(f"**:white_large_square: RBCs:** {rbc_count}")
        st.markdown(f"**:red_circle: WBCs:** {wbc_count}")
        st.markdown(f"**:black_circle: Platelets:** {platelet_count}")

        # === Enhanced WBC Classification ===
        if wbc_count > 0:
            with st.status("WBC Classification"):
                if st.session_state.wbc_classifications is None:
                    with st.spinner("🧠 Classifying WBC subtypes..."):
                        cropped_wbcs = crop_cells(
                                st.session_state.uploaded_image,
                                st.session_state.detection_result,
                                cell_class_id=1,
                                output_size=224
                        )
                        st.session_state.wbc_classifications = [
                                classify_image(crop) for crop in cropped_wbcs
                        ]

                wbc_preds = st.session_state.wbc_classifications
                df_wbc = pd.DataFrame(wbc_preds)

                # Emoji mapping (customize based on your model's labels)
                EMOJI_MAP = {
                        "Neutrophil": "🛡️",
                        "Lymphocyte": "🛡️",
                        "Monocyte": "🧍",
                        "Eosinophil": "🔴",
                        "Basophil": "🔵",
                        # Add more as needed
                }

                if "class" in df_wbc.columns:
                    # Bar chart summary
                    class_counts = df_wbc["class"].value_counts().sort_index()
                    st.bar_chart(class_counts)

                    # Dominant type summary
                    dominant = class_counts.idxmax()
                    st.success(f"🩺 **Dominant WBC:** {dominant} ({class_counts[dominant]} cells)")

    # === Left Column: Image & Full Detection Table ===
    with col1:
        st.subheader("🖼️ Annotated Image")

        # Plot filtered results
        result_copy = copy.deepcopy(st.session_state.detection_result)
        if class_to_plot != "All" and result_copy.boxes is not None:
            name_to_id = {"RBC": 0, "WBC": 1, "Platelet": 2}
            target_id = name_to_id[class_to_plot]
            mask = result_copy.boxes.cls == target_id
            from ultralytics.engine.results import Boxes
            result_copy.boxes = Boxes(
                    result_copy.boxes.data[mask],
                    result_copy.orig_shape
            )

        annotated_bgr = result_copy.plot()
        annotated_rgb = annotated_bgr[..., ::-1]  # BGR → RGB

        caption = f"Detected cells ({class_to_plot} only)" if class_to_plot != "All" else "All detected cells"
        st.image(annotated_rgb, caption=caption, use_container_width=True)

        # Full detection table
        st.subheader("📋 Full Detection Details")
        if detections:
            df_all = pd.DataFrame(detections)
            st.dataframe(df_all, use_container_width=True)
            st.success(f"✅ Detected **{len(detections)}** cells in total.")
        else:
            st.warning("No cells detected above the confidence threshold.")

else:
    # No results yet
    if uploaded_file is not None:
        st.info("👆 Click **'Detect & Classify Cells'** to start analysis.")
    else:
        st.info("📤 Please upload a blood smear image to begin.")
