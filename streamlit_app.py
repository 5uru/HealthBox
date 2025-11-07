import streamlit as st
from PIL import Image
import pandas as pd
import copy
from ultralytics.engine.results import Boxes

from healthbox.blood_stain.blood_analysis import run_blood_analysis
from healthbox.blood_stain.registry import OPTIONAL_TASKS


st.set_page_config(
        page_title="🩸 Blood Cell Detection & Classification",
        page_icon="🩸",
        layout="wide",
        initial_sidebar_state="expanded"
)


# Initialize session state
for key in ['uploaded_image', 'results', 'confidence_threshold', 'confidence_show']:
    if key not in st.session_state:
        st.session_state[key] = None

if st.session_state.confidence_threshold is None:
    st.session_state.confidence_threshold = 0.25
    st.session_state.confidence_show = False

for task_key in OPTIONAL_TASKS:
    if f"{task_key}_enabled" not in st.session_state:
        st.session_state[f"{task_key}_enabled"] = True


@st.dialog("Model Settings")
def model_setting():
    # Confidence threshold
    confidence_threshold = st.slider(
            "Confidence Threshold",
            min_value=0.0,
            max_value=1.0,
            value=st.session_state.confidence_threshold,
            step=0.05,
            help="Lower = more detections (may include false positives)"
    )

    confidence_show = st.checkbox(
            "Show Confidence Scores on Output Image",
            value=st.session_state.confidence_show,
            key="dialog_confidence_show"
    )
    with st.popover("### 🔍 Optional Analysis Tasks"):

        # Checkboxes pour chaque tâche
        enabled_tasks = {}
        for task_key, task_label in OPTIONAL_TASKS.items():
            enabled = st.checkbox(
                    task_label,
                    value=st.session_state[f"{task_key}_enabled"],
                    key=f"dialog_{task_key}"
            )
            enabled_tasks[task_key] = enabled

    if st.button("Apply Settings"):
        st.session_state.confidence_threshold = confidence_threshold
        st.session_state.confidence_show = confidence_show
        for task_key, enabled in enabled_tasks.items():
            st.session_state[f"{task_key}_enabled"] = enabled
        st.rerun()



col_title, col_setting = st.columns([4, 1])
with col_title:
    st.title("🩸 Blood Cell Detection & Classification")
with col_setting:
    if st.button("Model Settings"):
        model_setting()

st.markdown(
        "Upload a blood smear image to detect RBCs, WBCs, and Platelets, "
        "then classify WBC subtypes and screen RBCs for malaria."
)


uploaded_file = st.file_uploader("Choose an image...", type=['jpg', 'jpeg', 'png'])

if uploaded_file and st.button("🔬 Detect & Classify Cells"):
    try:
        image = Image.open(uploaded_file).convert("RGB")
        st.session_state.uploaded_image = image

        enabled_tasks = {
                task_key: st.session_state[f"{task_key}_enabled"]
                for task_key in OPTIONAL_TASKS
        }

        with st.spinner("🔍 Analyzing blood cells..."):
            # ✅ Delegate all heavy lifting to your blood_analysis module
            st.session_state.results = run_blood_analysis(
                    image,
                    conf_threshold=st.session_state.confidence_threshold,
                    enabled_tasks=enabled_tasks
            )
    except Exception as e:
        st.error(f"⚠️ Error during analysis: {str(e)}")

        st.session_state.results = None


if st.session_state.results is not None:
    res = st.session_state.results
    detections = res["detection"]["detections"]
    classification_results = res.get("classification", {})


    col1, col2 = st.columns([2, 1])

    with col2:
        st.header("📊 Results & Controls")

        class_to_plot = st.selectbox(
                "🔍 Highlight cell type:",
                ("All", "RBC", "WBC", "Platelet"),
                index=0
        )

        # Counts
        counts = {
            "RBC": sum(d["Class"] == "RBC" for d in detections),
            "WBC": sum(d["Class"] == "WBC" for d in detections),
            "Platelet": sum(d["Class"] == "Platelet" for d in detections),
        }
        total = len(detections)
        st.markdown(f"**Total Cells:** {total}")
        st.markdown(f"**:white_large_square: RBCs:** {counts['RBC']}")
        st.markdown(f"**:red_circle: WBCs:** {counts['WBC']}")
        st.markdown(f"**:black_circle: Platelets:** {counts['Platelet']}")

        # Classification Results

        for key in classification_results.keys():
            with st.expander(classification_results[key]["name"]):
                df_class = pd.DataFrame(classification_results[key]["results"])
                if "class" in df_class.columns:
                    class_counts = df_class["class"].value_counts().sort_index()
                    st.bar_chart(class_counts)
                st.info(classification_results[key]["info"])

    with col1:
        st.subheader("🖼️ Annotated Image")
        result_copy = copy.deepcopy(res["detection"]["result"])
        if class_to_plot != "All" and result_copy.boxes is not None:
            name_to_id = {"RBC": 0, "WBC": 1, "Platelet": 2}
            target_id = name_to_id[class_to_plot]
            mask = result_copy.boxes.cls == target_id

            result_copy.boxes = Boxes(result_copy.boxes.data[mask], result_copy.orig_shape)

        annotated_bgr = result_copy.plot(conf=st.session_state.confidence_show)
        annotated_rgb = annotated_bgr[..., ::-1]  # BGR to RGB
        caption = f"Detected ({class_to_plot})" if class_to_plot != "All" else "All detected cells"
        st.image(annotated_rgb, caption=caption, width='stretch')

else:
    if uploaded_file:
        st.info("👆 Click **'Detect & Classify Cells'** to start analysis.")
    else:
        st.info("📤 Please upload a blood smear image to begin.")