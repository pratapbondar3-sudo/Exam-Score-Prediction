import streamlit as st
import pandas as pd
import numpy as np
import joblib
import time

# Page configuration
st.set_page_config(
    page_title="Predictive Analytics Studio",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS styling for modern layout and buttons
st.markdown(
    """
    <style>
    .main {
        background-color: #fafafa;
    }
    .metric-card {
        background-color: #ffffff;
        border-radius: 12px;
        padding: 24px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
        border-left: 5px solid #2563eb;
        margin-top: 15px;
    }
    .stButton>button {
        background: linear-gradient(90deg, #2563eb, #1d4ed8);
        color: white;
        font-weight: 600;
        border-radius: 8px;
        padding: 0.6rem 1.8rem;
        border: none;
        transition: all 0.2s ease-in-out;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 15px rgba(37, 99, 235, 0.3);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Cache model loading for fast page reloads
@st.cache_resource
def load_model():
    try:
        model = joblib.load("model.pkl")
        return model
    except Exception as e:
        st.error(f"Error loading model.pkl: {e}")
        return None

model = load_model()

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/clouds/200/artificial-intelligence.png", width=120)
    st.title("Model Control Panel")
    st.info("Ensure input features correspond directly to your training schema.")
    
    st.markdown("---")
    effect_type = st.selectbox(
        "Celebration Effect",
        ["Balloons 🎉", "Snow ❄️", "Toast Only ⚡"],
        index=0
    )
    st.caption("Triggered upon successful prediction inference.")

# Header
st.title("⚡ AI Predictive Engine")
st.write("Enter parameter values below and generate dynamic real-time model inferences.")
st.markdown("---")

# Input Form organized in columns
st.subheader("Input Attributes")

col1, col2, col3 = st.columns(3)

with col1:
    feature_1 = st.number_input("Feature 1 (e.g., Age / Metric A)", value=30.0, step=1.0)
    feature_2 = st.number_input("Feature 2 (e.g., Balance / Metric B)", value=1500.0, step=50.0)

with col2:
    feature_3 = st.number_input("Feature 3 (e.g., Score / Metric C)", value=0.75, step=0.05)
    feature_4 = st.slider("Feature 4 (Threshold Scale)", min_value=0, max_value=100, value=45)

with col3:
    category = st.selectbox("Categorical Factor", ["Low Risk", "Medium Risk", "High Risk"])
    # Map categorical if your model requires numeric encoding:
    cat_mapping = {"Low Risk": 0, "Medium Risk": 1, "High Risk": 2}
    feature_5 = cat_mapping[category]
    st.metric("Encoded Factor Value", feature_5)

# Assemble DataFrame matching training feature columns
input_df = pd.DataFrame([[feature_1, feature_2, feature_3, feature_4, feature_5]],
                        columns=["Feature_1", "Feature_2", "Feature_3", "Feature_4", "Feature_5"])

st.markdown("### Verification Table")
st.dataframe(input_df, use_container_width=True)

# Prediction Action
predict_col, _ = st.columns([1, 4])
with predict_col:
    predict_clicked = st.button("Run Prediction 🚀", use_container_width=True)

if predict_clicked:
    if model is None:
        st.warning("Cannot run inference: `model.pkl` is missing or corrupted.")
    else:
        with st.spinner("Processing through inference pipeline..."):
            time.sleep(0.6)  # Brief visual delay for realistic execution feel
            
            # Run prediction
            try:
                # If model expects 2D numpy array instead of DataFrame: input_df.values
                prediction = model.predict(input_df.values)[0]
                
                # Check for probability prediction (if classifier)
                prob = None
                if hasattr(model, "predict_proba"):
                    prob = np.max(model.predict_proba(input_df.values)[0]) * 100
                
                # Trigger configured celebration effect
                if effect_type == "Balloons 🎉":
                    st.balloons()
                elif effect_type == "Snow ❄️":
                    st.snow()
                
                st.toast("Inference complete!", icon="✅")

                # Attractive Results Card
                st.markdown(
                    f"""
                    <div class="metric-card">
                        <h4 style="margin: 0; color: #6b7280; font-size: 0.95rem; text-transform: uppercase; letter-spacing: 0.05em;">Inference Result</h4>
                        <h1 style="margin: 8px 0 0 0; color: #1e293b; font-size: 2.2rem;">{prediction}</h1>
                        <p style="margin: 4px 0 0 0; color: #16a34a; font-weight: 500;">
                            {"Confidence Score: " + f"{prob:.2f}%" if prob is not None else "Model prediction produced successfully"}
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                
            except Exception as err:
                st.error(f"Inference failed: {err}")
