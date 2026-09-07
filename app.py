import streamlit as st
import pandas as pd
import numpy as np
import joblib
import time

# --- Page Settings ---
st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Modern Dashboard Styling ---
st.markdown(
    """
    <style>
    .main {
        background-color: #f8fafc;
    }
    .result-card {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border-radius: 16px;
        padding: 24px;
        color: white;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.1), 0 8px 10px -6px rgba(0, 0, 0, 0.1);
        border: 1px solid #334155;
        margin-top: 20px;
    }
    .score-badge {
        font-size: 3rem;
        font-weight: 800;
        background: linear-gradient(90deg, #38bdf8, #818cf8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
    }
    div.stButton > button:first-child {
        background: linear-gradient(90deg, #3b82f6 0%, #2563eb 100%);
        color: white;
        font-weight: 600;
        font-size: 1rem;
        border-radius: 10px;
        padding: 0.65rem 2rem;
        border: none;
        box-shadow: 0 4px 14px rgba(37, 99, 235, 0.35);
        transition: all 0.2s ease-in-out;
    }
    div.stButton > button:first-child:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(37, 99, 235, 0.45);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# --- Model Loading ---
@st.cache_resource
def load_svm_model():
    try:
        # Attempts to load svm.pkl, fallback to model.pkl if renamed
        return joblib.load("svm.pkl")
    except FileNotFoundError:
        try:
            return joblib.load("model.pkl")
        except Exception as e:
            st.error(f"Could not find 'svm.pkl': {e}")
            return None
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_svm_model()

# --- Sidebar Controls ---
with st.sidebar:
    st.header("⚙️ App Settings")
    celebration_fx = st.selectbox(
        "Prediction Effect",
        ["Balloons 🎉", "Snow ❄️", "Toast Notification ⚡"],
        index=0,
    )
    st.markdown("---")
    st.markdown(
        """
        **Model Details:**
        - **Algorithm:** Support Vector Regressor (`SVR`)
        - **Kernel:** RBF (Radial Basis Function)
        - **Input Features:** 11 Parameters
        """
    )

# --- App Header ---
st.title("🎓 Student Performance Forecasting")
st.caption("Predict estimated exam performance based on academic, lifestyle, and institutional factors.")
st.markdown("---")

# --- Structured Inputs ---
with st.container():
    st.subheader("1. Demographics & Course")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        age = st.number_input("Age", min_value=15, max_value=60, value=21, step=1)
    with col2:
        gender_label = st.selectbox("Gender", ["Female", "Male", "Other"])
        gender_map = {"Female": 0, "Male": 1, "Other": 2}
        gender = gender_map[gender_label]
    with col3:
        course_label = st.selectbox("Course Track", ["Computer Science", "Engineering", "Business", "Arts & Sciences"])
        course_map = {"Computer Science": 0, "Engineering": 1, "Business": 2, "Arts & Sciences": 3}
        course = course_map[course_label]

st.markdown("---")

with st.container():
    st.subheader("2. Study Habits & Attendance")
    col4, col5, col6 = st.columns(3)
    
    with col4:
        study_hours = st.slider("Weekly Study Hours", min_value=0.0, max_value=60.0, value=18.5, step=0.5)
    with col5:
        class_attendance = st.slider("Class Attendance Rate (%)", min_value=0.0, max_value=100.0, value=85.0, step=1.0)
    with col6:
        study_method_label = st.selectbox("Primary Study Method", ["Self-Study", "Group Study", "Online Tutorials", "Coaching"])
        study_method_map = {"Self-Study": 0, "Group Study": 1, "Online Tutorials": 2, "Coaching": 3}
        study_method = study_method_map[study_method_label]

st.markdown("---")

with st.container():
    st.subheader("3. Environment, Wellness & Exam Parameters")
    col7, col8, col9 = st.columns(3)
    
    with col7:
        internet_access_label = st.radio("Internet Access at Home", ["Yes", "No"], horizontal=True)
        internet_access = 1 if internet_access_label == "Yes" else 0
        sleep_hours = st.number_input("Average Sleep Hours / Day", min_value=3.0, max_value=12.0, value=7.0, step=0.5)
    with col8:
        sleep_quality = st.slider("Sleep Quality Rating (1 = Poor, 5 = Excellent)", min_value=1, max_value=5, value=4)
        facility_rating = st.slider("Campus Facility Rating (1 = Basic, 5 = Premium)", min_value=1, max_value=5, value=3)
    with col9:
        exam_difficulty_label = st.selectbox("Exam Difficulty Level", ["Easy", "Moderate", "Hard"])
        diff_map = {"Easy": 1, "Moderate": 2, "Hard": 3}
        exam_difficulty = diff_map[exam_difficulty_label]

# --- Assemble DataFrame matching training feature names ---
feature_columns = [
    "age",
    "gender",
    "course",
    "study_hours",
    "class_attendance",
    "internet_access",
    "sleep_hours",
    "sleep_quality",
    "study_method",
    "facility_rating",
    "exam_difficulty",
]

input_data = pd.DataFrame(
    [[
        float(age),
        float(gender),
        float(course),
        float(study_hours),
        float(class_attendance),
        float(internet_access),
        float(sleep_hours),
        float(sleep_quality),
        float(study_method),
        float(facility_rating),
        float(exam_difficulty),
    ]],
    columns=feature_columns,
)

st.markdown("---")

# --- Run Inference ---
btn_col, _ = st.columns([1, 4])
with btn_col:
    predict_btn = st.button("Generate Score Prediction 🚀", use_container_width=True)

if predict_btn:
    if model is None:
        st.error("Model file `svm.pkl` was not loaded properly.")
    else:
        with st.spinner("Executing SVR kernel projection..."):
            time.sleep(0.5)  # brief UI pause for smoother feel
            
            try:
                prediction = model.predict(input_data)[0]

                # Trigger selected effect
                if celebration_fx == "Balloons 🎉":
                    st.balloons()
                elif celebration_fx == "Snow ❄️":
                    st.snow()
                st.toast("Inference generated successfully!", icon="✅")

                # Attractive Output Display Card
                st.markdown(
                    f"""
                    <div class="result-card">
                        <div style="font-size: 0.85rem; letter-spacing: 0.1em; text-transform: uppercase; color: #94a3b8;">
                            Estimated Academic Score
                        </div>
                        <p class="score-badge">{prediction:.2f}</p>
                        <p style="color: #cbd5e1; margin-top: 8px; font-size: 0.95rem;">
                            Model: <b>SVR (RBF Kernel)</b> | Features: <b>11 evaluated attributes</b>
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            except Exception as ex:
                st.error(f"Inference execution failed: {ex}")
