import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px # CORRECTED: Changed 'ploty.express' to 'plotly.express'
from sklearn.preprocessing import LabelEncoder
# NOTE: The actual model (e.g., Keras/TensorFlow) trained in your
# welcome_to_colab.py would be loaded here. For this demo,
# the prediction is based on a simulated heuristic.

# --- 1. CONFIGURATION AND STYLING (The "Crazy" UI) ---

st.set_page_config(
    page_title="Diab-AI: Predictive Health Dashboard",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for the "Crazy" UI - Dark, Neon, and Sharp
# This injects a high-contrast, dark-mode theme with vibrant accents.
st.markdown("""
<style>
    /* Main body background and text */
    .main {
        background-color: #0d0c1d; /* Deep Space Blue/Black */
        color: #e6e6ff; /* Off-white for text */
    }

    /* Sidebar background and styling */
    .css-1d391kg { /* Targetting the sidebar container */
        background-color: #1a1a3a; /* Darker blue/purple for sidebar */
        border-right: 3px solid #00F0FF; /* Neon Cyan border */
        box-shadow: 2px 0 10px rgba(0, 240, 255, 0.5);
    }
    
    /* Header/Title Styling */
    .css-1n74xts { /* Targetting st.title */
        color: #FF00FF !important; /* Neon Pink */
        font-family: 'Space Mono', monospace;
        text-shadow: 0 0 5px #FF00FF, 0 0 10px #FF00FF;
    }
    
    /* Custom button styling (Neon Green) */
    div.stButton > button:first-child {
        background-color: #00FF00;
        color: black;
        border: 2px solid #00FF00;
        box-shadow: 0 0 8px #00FF00, 0 0 15px #00FF00;
        transition: all 0.3s;
        font-weight: bold;
        border-radius: 12px;
    }
    div.stButton > button:first-child:hover {
        background-color: #00F0FF;
        color: black;
        border-color: #00F0FF;
        box-shadow: 0 0 15px #00F0FF, 0 0 25px #00F0FF;
    }
    
    /* Customized Metric Boxes (Neon Blue) */
    [data-testid="stMetric"] > div {
        background-color: #1a1a3a;
        padding: 15px;
        border-radius: 15px;
        border: 1px solid #00F0FF;
        box-shadow: 0 0 10px rgba(0, 240, 255, 0.4);
    }
    
    /* Tabs styling (making active tab stand out with neon pink) */
    .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
        font-size: 1.1rem;
        font-weight: bold;
    }
    .stTabs [data-baseweb="tab-list"] button:focus {
        border-bottom: 3px solid #FF00FF !important;
        color: #FF00FF !important;
    }

</style>
""", unsafe_allow_html=True)


# --- 2. DATA LOADING AND PREPROCESSING ---

@st.cache_data
def load_data():
    """Loads and preprocesses the dataset."""
    try:
        # Load the data file provided by the user
        df = pd.read_csv("diabetes_young_adults_india.csv")
    except FileNotFoundError:
        st.error("Error: 'diabetes_young_adults_india.csv' not found. Please ensure the file is in the correct path.")
        return pd.DataFrame()

    # Apply minimal preprocessing
    if 'ID' in df.columns:
        df.drop('ID', axis=1, inplace=True)
    df.fillna(method='ffill', inplace=True) # Fill NaNs
    
    # Simple Label Encoding for target variable for analysis visualization
    if 'Prediabetes' in df.columns:
        le = LabelEncoder()
        df['Prediabetes_Encoded'] = le.fit_transform(df['Prediabetes'])
    
    return df

df = load_data()

# --- 3. UI LAYOUT ---

st.title("🧬 Diab-AI: Predictive Health Dashboard")
st.caption("Neuro-Symphony: An ML-Powered Analysis of Diabetes Factors in Young Indian Adults.")

if df.empty:
    st.stop()

# --- 4. SIDEBAR FILTERS ---
with st.sidebar:
    st.header("Global Filters ⚙️")
    
    # Age Filter
    age_min = int(df['Age'].min())
    age_max = int(df['Age'].max())
    age_range = st.slider(
        'Select Age Range',
        min_value=age_min,
        max_value=age_max,
        value=(age_min, age_max)
    )
    
    # Region Filter
    regions = df['Region'].unique()
    selected_regions = st.multiselect(
        'Select Region(s)',
        options=regions,
        default=regions
    )
    
    # Apply global filter
    df_filtered = df[
        (df['Age'] >= age_range[0]) & 
        (df['Age'] <= age_range[1]) &
        (df['Region'].isin(selected_regions))
    ]
    
    st.info(f"Showing **{len(df_filtered)}** of **{len(df)}** records.")

# --- 5. TABS FOR DIFFERENT VIEWS ---
tab1, tab2, tab3 = st.tabs(["🚀 Data Overview & Metrics", "📊 Deep Dive Analysis", "🧠 Prediction Simulator"])

# --- TAB 1: DATA OVERVIEW & METRICS ---
with tab1:
    st.header("Snapshot Metrics")
    
    # Create Metrics using a 4-column layout
    col1, col2, col3, col4 = st.columns(4)
    
    total_records = len(df_filtered)
    prediabetes_count = df_filtered['Prediabetes'].eq('Yes').sum()
    avg_bmi = df_filtered['BMI'].mean()
    avg_hba1c = df_filtered['HbA1c'].mean()
    
    col1.metric("Total Records Analyzed", f"{total_records:,}", delta=f"Filtered from {len(df):,}")
    col2.metric("Prediabetes Cases", f"{prediabetes_count:,}", f"{prediabetes_count / total_records * 100:.1f}% of filtered data")
    col3.metric("Avg BMI (Filtered)", f"{avg_bmi:.2f} kg/m²", delta="Normal range ~18.5-24.9")
    col4.metric("Avg HbA1c (Filtered)", f"{avg_hba1c:.2f}%", delta="Normal range <5.7%")
    
    st.divider()

    st.header("Raw Data Preview")
    st.dataframe(df_filtered.head(10), use_container_width=True)

# --- TAB 2: INTERACTIVE ANALYSIS ---
with tab2:
    st.header("Interactive Visualizations")
    
    # Row 1: Distribution Plot
    st.subheader("Distribution Viewer")
    col1_2, col2_2 = st.columns([1, 2])
    
    with col1_2:
        dist_col = st.selectbox(
            "Select Column for Distribution",
            options=['Fasting_Blood_Sugar', 'BMI', 'HbA1c', 'Cholesterol_Level', 'Sleep_Hours', 'Screen_Time']
        )
    
    with col2_2:
        fig_hist = px.histogram(
            df_filtered, 
            x=dist_col, 
            color='Gender',
            title=f"Distribution of {dist_col} by Gender",
            template="plotly_dark",
            color_discrete_map={'Male': '#00F0FF', 'Female': '#FF00FF'}
        )
        fig_hist.update_layout(height=400)
        st.plotly_chart(fig_hist, use_container_width=True)

    st.divider()
    
# --- TAB 3: PREDICTION SIMULATOR ---
with tab3:
    st.header("Patient Outcome Simulation 🧠")
    st.write("Input the parameters below to simulate the likelihood of Prediabetes, based on the factors used in our Neural Network model.")
    
    # Prediction Input Form Layout (3 columns)
    input_cols = st.columns(3)

    with input_cols[0]:
        st.subheader("Physical/Social")
        sim_age = st.slider("Age", 17, 25, 21)
        sim_gender = st.radio("Gender", ['Male', 'Female'])
        sim_bmi = st.number_input("BMI (kg/m²)", min_value=15.0, max_value=50.0, value=28.5, step=0.1)
        sim_activity = st.selectbox("Physical Activity", ['Sedentary', 'Moderate', 'Active'])
        
    with input_cols[1]:
        st.subheader("Biomarkers")
        sim_fbs = st.number_input("Fasting Blood Sugar (mg/dL)", min_value=70.0, max_value=200.0, value=115.0, step=0.1)
        sim_hba1c = st.number_input("HbA1c (%)", min_value=4.0, max_value=12.0, value=6.2, step=0.1)
        sim_cholesterol = st.number_input("Cholesterol Level", min_value=100.0, max_value=400.0, value=210.0, step=0.1)
        sim_sleep = st.slider("Sleep Hours", 3.0, 10.0, 7.5, step=0.1)
        
    with input_cols[2]:
        st.subheader("Lifestyle/Family")
        sim_screen = st.slider("Screen Time (hrs/day)", 1.0, 15.0, 8.0, step=0.1)
        sim_stress = st.slider("Stress Level (1-10)", 1, 10, 6)
        sim_smoking = st.checkbox("Smoker?", False)
        sim_family_hist = st.checkbox("Family History of Diabetes?", True)
        
    st.divider()
    
    # Prediction Button
    if st.button("RUN NEURO-PREDICTION"):
        
        # --- MOCK PREDICTION LOGIC (Placeholder for NN Model) ---
        # This simulates a prediction based on some risk factors. 
        # In a real application, you would load your trained Keras model here.
        risk_score = 0
        if sim_hba1c > 6.5:
            risk_score += 0.4
        if sim_bmi > 30 and sim_fbs > 120:
            risk_score += 0.3
        if sim_family_hist:
            risk_score += 0.2
            
        # Add random noise for a dynamic feel
        risk_score = np.clip(risk_score + np.random.uniform(-0.1, 0.1), 0.05, 0.95)
        
        # Result Display (2 columns)
        result_col1, result_col2 = st.columns([1, 2])
        
        prediction_status = "HIGH RISK (Prediabetes Likely)" if risk_score >= 0.5 else "LOW RISK (Healthy Profile)"
        
        with result_col1:
            st.markdown(f"### **Prediction Outcome**")
            
            if risk_score >= 0.5:
                st.error(f"🔴 {prediction_status}", icon="⚠️")
                st.markdown(f"**Recommendation:** Consult a doctor and focus on diet and exercise.")
            else:
                st.success(f"🟢 {prediction_status}", icon="✅")
                st.markdown(f"**Recommendation:** Maintain current healthy lifestyle and monitor biomarkers.")

        with result_col2:
            st.markdown(f"### **Confidence Score**")
            st.progress(risk_score)
            st.metric("Prediabetes Likelihood", f"{risk_score * 100:.1f}%")
            
            st.caption("""
            *Disclaimer: This is a simulated prediction. For health decisions, 
            always consult a qualified healthcare professional.*
            """)
