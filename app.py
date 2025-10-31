import streamlit as st
import pandas as pd
import numpy as np
import time # For simulation of prediction time
import json # To store and display mock model metadata

# --- 1. PAGE CONFIGURATION AND CUSTOM NEON STYLES ---
st.set_page_config(
    page_title="NEURO-DIABETES PREDICTIVE ANALYZER",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for the "Crazy" UI - Neon Glow Theme
custom_css = """
<style>
    /* Global Background and Text */
    .stApp {
        background: linear-gradient(135deg, #1f012b 0%, #000000 70%);
        color: #ffffff;
    }

    /* Main Title - Extreme Neon Glow */
    h1 {
        text-align: center;
        font-family: 'Arial Black', Gadget, sans-serif;
        font-size: 3.5em;
        text-shadow:
            0 0 5px #00ffff,
            0 0 10px #00ffff,
            0 0 20px #00ffff,
            0 0 40px #ff00ff,
            0 0 80px #ff00ff,
            0 0 90px #ff00ff,
            0 0 100px #ff00ff,
            0 0 150px #ff00ff;
        color: #fff;
        padding-top: 20px;
        padding-bottom: 20px;
    }

    /* Sidebar Styling */
    /* Targeting the main sidebar containers */
    .css-1d3w5oq, .css-6qob1n, .css-1dp549u {
        background-color: #0d0115 !important;
        border-right: 3px solid #00ffff !important;
        box-shadow: 5px 0 10px rgba(0, 255, 255, 0.5);
    }

    /* Input/Widget Styling - Futuristic look */
    .stTextInput>div>div>input, .stSlider>div>div>div, .stNumberInput>div>div>input {
        background-color: #2b0340 !important;
        border: 1px solid #ff00ff !important;
        color: #00ffff !important;
        box-shadow: 0 0 5px #ff00ff;
        padding: 10px;
        border-radius: 8px;
    }

    /* Button Styling - Extreme */
    .stButton>button {
        background: #00ffff;
        color: #1f012b;
        font-weight: bold;
        border: 2px solid #ff00ff;
        border-radius: 12px;
        padding: 10px 20px;
        box-shadow: 0 0 15px #00ffff, 0 0 25px #ff00ff;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        background: #ff00ff;
        color: #00ffff;
        box-shadow: 0 0 20px #ff00ff, 0 0 35px #00ffff;
        transform: scale(1.05);
    }

    /* Prediction Card Style (for the st.container) */
    .prediction-card {
        padding: 30px;
        border-radius: 20px;
        background: rgba(43, 3, 64, 0.7); /* Darker purple transparent */
        border: 5px solid;
        border-image: linear-gradient(45deg, #00ffff, #ff00ff) 1;
        box-shadow: 0 0 30px rgba(255, 0, 255, 0.8), 0 0 60px rgba(0, 255, 255, 0.5);
        text-align: center;
        margin-top: 30px;
    }

    /* Metric/Result Styling - Targeting the value of st.metric */
    div[data-testid="stMetricValue"] {
        font-size: 3em;
        color: #00ffaa; /* Green Neon */
        text-shadow: 0 0 10px #00ffaa;
    }

    /* Data Review Card for Input Display */
    .data-card {
        padding: 15px;
        border-left: 5px solid #00ffaa;
        background-color: rgba(31, 1, 43, 0.8);
        border-radius: 8px;
        margin-bottom: 15px;
    }
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

# --- 2. MOCK DATA & ML LOGIC ---

# Define feature boundaries for the UI, inspired by your ML pipeline data
FEATURE_DEFS = {
    "Age": {"min": 18, "max": 75, "default": 35, "help": "Patient age in years."},
    "BMI": {"min": 15.0, "max": 50.0, "default": 25.0, "help": "Body Mass Index (kg/m²)."},
    "Genetic_Risk_Score": {"min": 0.0, "max": 1.0, "default": 0.5, "help": "Synthesized risk score from genetic markers (0=Low, 1=High)."},
    "HbA1c": {"min": 4.0, "max": 15.0, "default": 5.7, "help": "Average blood sugar levels over the past 3 months (%)."},
    "Fasting_Blood_Sugar": {"min": 60.0, "max": 300.0, "default": 100.0, "help": "Glucose level after 8 hours of fasting (mg/dL)."},
    "Physical_Activity_Score": {"min": 0, "max": 10, "default": 5, "help": "Patient reported activity level (0=Sedentary, 10=Very Active)."},
}

MOCK_MODEL_METADATA = {
    "Model Name": "RandomForestClassifier v2.1 (Tuned)",
    "Accuracy": "94.7%",
    "Training Date": "2025-10-31",
    "Features Used": list(FEATURE_DEFS.keys()),
    "Target": "Diabetes_Type (0/1)"
}

def mock_prediction_function(data):
    """
    Simulates a sophisticated ML prediction based on input data.
    The higher the risk factors, the higher the prediction probability for Diabetes.
    """
    # Simple risk calculation: sum of scaled risk factors
    risk_score = (
        data['BMI'] * 0.15 +
        data['Genetic_Risk_Score'] * 0.40 + # High weight
        data['HbA1c'] * 0.20 +
        data['Fasting_Blood_Sugar'] * 0.15 +
        (10 - data['Physical_Activity_Score']) * 0.10 # Inverse weight for activity
    )

    # Normalize the score to a probability (0 to 1)
    probability = np.clip((risk_score / 150) * 1.5, 0.05, 0.95)

    # Simulate Random Forest's ensemble nature by adding small noise
    probability += np.random.uniform(-0.02, 0.02)
    probability = np.clip(probability, 0.01, 0.99)

    # Classification based on a 50% threshold
    prediction = 1 if probability >= 0.5 else 0

    # Ensure high risk factors result in high probability for dramatic effect
    if data['HbA1c'] >= 6.5 and data['Fasting_Blood_Sugar'] >= 126:
        probability = np.clip(probability * 1.2, 0.7, 0.99)
        prediction = 1

    return prediction, probability

# --- 3. SIDEBAR: USER INPUTS ---

with st.sidebar:
    st.title("🔬 NEURO-SCAN INPUT MODULE")
    st.markdown("---")
    st.header("Patient Biometrics")

    # Use a form to group inputs and handle submission
    with st.form(key='prediction_form'):
        input_data = {}

        # Dynamically create input widgets for features
        for feature, props in FEATURE_DEFS.items():
            if feature in ["Age", "Physical_Activity_Score"]:
                input_data[feature] = st.slider(
                    f"{feature.replace('_', ' ')}",
                    props["min"], props["max"], props["default"],
                    help=props["help"]
                )
            else:
                input_data[feature] = st.number_input(
                    f"{feature.replace('_', ' ')}",
                    props["min"], props["max"], props["default"],
                    step=0.1,
                    help=props["help"]
                )

        st.markdown("---")
        submit_button = st.form_submit_button(label='⚡ RUN PREDICTIVE SCAN ⚡')

# --- 4. MAIN DASHBOARD ---

st.markdown("<h1>NEURO-DIABETES PREDICTIVE ANALYZER</h1>", unsafe_allow_html=True)
st.markdown("---")

if submit_button:
    # ------------------
    # A. Display Loading Animation (Crazy UI Element)
    # ------------------
    with st.container():
        st.markdown("<h2 style='text-align:center; color:#00ffff;'>INITIATING QUANTUM ANALYSIS...</h2>", unsafe_allow_html=True)
        
        progress_bar = st.progress(0)
        status_text = st.empty()

        for i in range(100):
            time.sleep(0.01) # Small delay for the animation
            progress_bar.progress(i + 1)
            if i < 30:
                status_text.text(f"Processing Biometrics... ({i}%)")
            elif i < 70:
                status_text.text(f"Applying Ensemble Model Logic... ({i}%)")
            else:
                status_text.text(f"Finalizing Prediction Matrix... ({i}%)")

        progress_bar.empty()
        status_text.empty()
        st.success("SCAN COMPLETE. RESULTS READY.")

    # ------------------
    # B. Run Model and Display Results
    # ------------------
    prediction, probability = mock_prediction_function(input_data)
    risk_percent = probability * 100

    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("Input Data Review")
        # Display inputs in the custom styled data-card
        for key, value in input_data.items():
            st.markdown(
                f"""
                <div class="data-card">
                    <p style='color:#fff; margin: 0; font-size: 1.1em;'>
                        <strong style='color:#ff00ff;'>{key.replace('_', ' ')}:</strong>
                        <span style='float:right;'>{value}</span>
                    </p>
                </div>
                """,
                unsafe_allow_html=True
            )
        st.subheader("Model Metadata")
        st.json(MOCK_MODEL_METADATA)

    with col2:
        st.subheader("Predictive Outcome")
        with st.container():
            # Apply the "prediction-card" class
            st.markdown('<div class="prediction-card">', unsafe_allow_html=True)

            if prediction == 1:
                st.markdown(
                    f"""
                    <h2 style='color:#ff4d4d; text-shadow: 0 0 10px #ff0000;'>
                        🚨 DIABETES RISK DETECTED 🚨
                    </h2>
                    <h3 style='color:#ff4d4d;'>
                        CLASSIFICATION: TYPE 2 DIABETES LIKELY
                    </h3>
                    """,
                    unsafe_allow_html=True
                )
                emoji = "🔴"
                color_code = "#ff4d4d"
                message = "Immediate clinical consultation advised."
            else:
                st.markdown(
                    f"""
                    <h2 style='color:#00ffaa; text-shadow: 0 0 10px #00ffaa;'>
                        ✅ LOW RISK STATUS ✅
                    </h2>
                    <h3 style='color:#00ffaa;'>
                        CLASSIFICATION: NO DIABETES LIKELY
                    </h3>
                    """,
                    unsafe_allow_html=True
                )
                emoji = "🟢"
                color_code = "#00ffaa"
                message = "Continue healthy lifestyle habits."

            # Separator with glowing effect
            st.markdown(
                f"""
                <div style='
                    background: {color_code}20;
                    border: 1px solid {color_code};
                    padding: 15px;
                    border-radius: 10px;
                    margin-top: 20px;
                '>
                    <h2 style='margin-bottom: 0; color: {color_code};'>
                        {emoji} PATIENT RISK PROBABILITY {emoji}
                    </h2>
                </div>
                """, unsafe_allow_html=True
            )

            # Use st.metric for large, dramatic number display
            st.metric(
                label="Probability of Diabetes (%)",
                value=f"{risk_percent:.1f}%",
                delta=f"{probability:.2f} (Model Confidence)",
                delta_color="off"
            )

            st.markdown(
                f"""
                <p style='font-size: 1.3em; margin-top: 20px; color:#ffffff;'>
                    <strong style='color:#00ffff;'>MODEL INSIGHT:</strong> {message}
                </p>
                """, unsafe_allow_html=True
            )

            st.markdown('</div>', unsafe_allow_html=True)

else:
    # ------------------
    # C. Initial Welcome State
    # ------------------
    st.info("👈 Enter patient data in the sidebar and click 'RUN PREDICTIVE SCAN' to analyze results using our highly accurate Neuro-Model!")
    st.image("https://placehold.co/800x400/1f012b/00ffff?text=Futuristic+Data+Visualization+Pending", use_column_width=True)
    st.markdown(
        """
        <div style="text-align: center; margin-top: 30px; padding: 20px; border: 1px dashed #ff00ff; border-radius: 10px;">
            <p style="font-size: 1.2em; color: #fff;">
                The **NEURO-DIABETES PREDICTIVE ANALYZER** uses an ensemble of algorithms
                (inspired by **Random Forest** and Deep Learning architectures) to process
                key biometric and genetic indicators. Our goal is to achieve **99% confidence**
                in early-stage risk detection.
            </p>
        </div>
        """, unsafe_allow_html=True
    )
