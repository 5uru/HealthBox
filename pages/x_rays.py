import streamlit as st
from PIL import Image
import pandas as pd
import copy
import numpy as np

from healthbox.x_rays.x_rays_analysis import run_x_rays_analysis
from healthbox.blood_stain.registry import OPTIONAL_TASKS

st.set_page_config(
        page_title="Chest XRay Analysis",
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
    st.title("Chest XRay Analysis")
with col_setting:
    if st.button("Model Settings"):
        model_setting()


cl1, cl2 = st.columns([1, 4])
with cl1:
    st.markdown(
            "Upload a chest xray image."
    )
    uploaded_file = st.file_uploader("Choose an image...", type=['jpg', 'jpeg', 'png'])
with cl2:
    st.markdown("Medical history and questioning")
    birthday = st.date_input("Patient birthday", value=pd.to_datetime("2000-01-01"))
    sex = st.selectbox("Patient sex", options=["Male", "Female"])
    weight = st.number_input("Patient weight (kg)", min_value=0.0, max_value=500.0, value=70.0)
    height = st.number_input("Patient height (cm)", min_value=0.0, max_value=300.0, value=170.0)
    smoking_status = st.selectbox(
            "Smoking status",
            options=["Non-smoker", "Former smoker", "Current smoker"]
    )
    pulse  = st.number_input("Pulse", min_value=0.0, max_value=300.0, value=80.0)
    blood_pressure = st.number_input("Blood pressure", min_value=0.0, max_value=300.0, value=120.0)
    tension = st.number_input("Tension", min_value=0.0, max_value=300.0, value=80.0)
    blood_sugar_level = st.number_input("Blood sugar level", min_value=0.0, max_value=1000.0, value=100.0)
    personal_background = st.text_input("Personal background")
    family_history = st.text_input("Family history")
    reason_for_exam = st.text_input("Reason for exam")


if uploaded_file and st.button("🔬 Detect & Classify Cells"):
    try:
        image = Image.open(uploaded_file).convert("RGB")
        st.session_state.uploaded_image = image


        enabled_tasks = {
                task_key: st.session_state[f"{task_key}_enabled"]
                for task_key in OPTIONAL_TASKS
        }

        with st.spinner("🔍 Analyzing chest xray..."):
            # ✅ Delegate all heavy lifting to your blood_analysis module
            st.session_state.results = run_x_rays_analysis(
                    image,
                    conf_threshold=st.session_state.confidence_threshold,
                    enabled_tasks=enabled_tasks
            )
    except Exception as e:
        st.error(f"⚠️ Error during analysis: {str(e)}")

        st.session_state.results = None


if st.session_state.results is not None:
    res = st.session_state.results
    detections = res["detection"]
    classification_results = res.get("classification", {})


    col1, col2 = st.columns([2, 1])

    with col2:
        st.header("Results & Controls")

        show_detections = []
        options = detections.keys()
        options_names = [detections[i]["name"] for i in options]
        selection = st.segmented_control(
                "Show Detections", options_names, selection_mode="single",default="Chest Segmentation"
        )
        for det in detections:
            if detections[det]["name"] == selection:
                show_detections.append(det)
        st.write("Show Detections:")
        for det in detections:
            st.markdown(res["detection"][det]["detections"])

        for key in classification_results:
            results = classification_results[key]["results"]
            st.markdown(results)




    with col1:
        st.subheader("Annotated Image")
        for det_key in res["detection"]:
            if not det_key in show_detections:
                continue

            result_copy = copy.deepcopy(res["detection"][det_key]["result"])

            st.image(result_copy, width='stretch')




else:
    if uploaded_file:
        st.info("👆 Click to start analysis.")
    else:
        st.info("📤 Please upload a chest xray image to begin.")