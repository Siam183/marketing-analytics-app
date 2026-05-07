import streamlit as st
import joblib
import pandas as pd
import numpy as np

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="AI Marketing Intelligence Dashboard",
    page_icon="🎯",
    layout="wide"
)

# --- CUSTOM CSS FOR VISIBILITY ---
# This CSS ensures the cards have a distinct background and the text is always visible
st.markdown("""
    <style>
    /* Card Container */
    [data-testid="stMetric"] {
        background-color: #ffffff !important;
        border: 1px solid #e0e0e0;
        padding: 15px !important;
        border-radius: 10px !important;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    /* Force Labels (Titles) to be Dark Gray */
    [data-testid="stMetricLabel"] {
        color: #555555 !important;
        font-weight: 600 !important;
    }
    /* Force Values (Numbers) to be Black */
    [data-testid="stMetricValue"] {
        color: #1a1a1a !important;
        font-weight: 700 !important;
    }
    /* Fix for Delta (the red/green numbers) */
    [data-testid="stMetricDelta"] {
        font-weight: 500 !important;
    }
    </style>
    """, unsafe_allow_html=True)

# --- LOAD ASSETS ---
@st.cache_resource
def load_all():
    return joblib.load('marketing_hybrid_model.pkl')

try:
    assets = load_all()
    lookup = assets['lookup_table']
    ml_model = assets['ml_model']
    encoders = assets['label_encoders']
    raw_df = assets['df']
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# --- HEADER ---
st.title("🎯 Hybrid Marketing Forecaster")
st.markdown("Comparing **Historical Reality** with **Machine Learning Predictions**.")
st.divider()

# --- DYNAMIC SIDEBAR FILTERS ---
with st.sidebar:
    st.header("Campaign Parameters")
    
    comp = st.selectbox("Company", sorted(raw_df['Company'].unique()))
    
    loc_options = sorted(raw_df[raw_df['Company'] == comp]['Location'].unique())
    loc = st.selectbox("Geographic Location", loc_options)
    
    chan_options = sorted(raw_df[(raw_df['Company'] == comp) & (raw_df['Location'] == loc)]['Channel_Used'].unique())
    chan = st.selectbox("Marketing Channel", chan_options)
    
    st.divider()
    
    aud = st.selectbox("Target Audience", sorted(raw_df['Target_Audience'].unique()))
    ctype = st.selectbox("Campaign Type", sorted(raw_df['Campaign_Type'].unique()))
    lang = st.selectbox("Language", sorted(raw_df['Language'].unique()))
    seg = st.selectbox("Customer Segment", sorted(raw_df['Customer_Segment'].unique()))

# --- MAIN LOGIC ---
if st.button("🚀 Run Intelligence Report", type="primary", use_container_width=True):
    
    # 1. HISTORICAL LOOKUP
    match = lookup[
        (lookup['Company'] == comp) & (lookup['Location'] == loc) & 
        (lookup['Channel_Used'] == chan) & (lookup['Target_Audience'] == aud) &
        (lookup['Campaign_Type'] == ctype) & (lookup['Language'] == lang) &
        (lookup['Customer_Segment'] == seg)
    ]

    # 2. AI PREDICTION
    input_row = pd.DataFrame([{
        'Company': comp, 'Campaign_Type': ctype, 'Target_Audience': aud,
        'Duration': 30, 'Channel_Used': chan, 'Location': loc,
        'Language': lang, 'Customer_Segment': seg
    }])
    
    try:
        for col, le in encoders.items():
            input_row[col] = le.transform(input_row[col])
        ml_pred = ml_model.predict(input_row)[0]
    except Exception as e:
        st.error(f"Prediction Error: {e}")
        ml_pred = [0, 0, 0]

    # --- RESULTS DISPLAY ---
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📊 Historical Data (Actuals)")
        if not match.empty:
            res = match.iloc[0]
            st.metric("Avg. ROI", f"{res['ROI']:.2f}x")
            st.metric("Avg. Conv. Rate", f"{res['Conversion_Rate']*100:.2f}%")
            st.metric("Avg. Acq. Cost", f"${res['Acquisition_Cost']:,.2f}")
        else:
            st.warning("⚠️ No exact historical match found.")

    with col2:
        st.subheader("🤖 AI Prediction (XGBoost)")
        roi_delta = ml_pred[2] - lookup['ROI'].mean()
        
        st.metric("Predicted ROI", f"{ml_pred[2]:.2f}x", delta=f"{roi_delta:.2f} vs Market Avg")
        st.metric("Predicted Conv. Rate", f"{ml_pred[0]*100:.2f}%")
        st.metric("Predicted Acq. Cost", f"${ml_pred[1]:,.2f}")

    # --- STRATEGIC INSIGHT ---
    st.divider()
    with st.expander("💡 Strategic Analysis", expanded=True):
        if ml_pred[2] > 3.0:
            st.success("**High Performance Expected:** Scalable configuration.")
        elif ml_pred[2] < 1.5:
            st.error("**Caution:** Predicted ROI is below target.")
        else:
            st.info("**Steady Performance:** Yielding average market returns.")

else:
    st.info("👈 Select your parameters and click 'Run' to see results.")

st.markdown("---")
st.caption("Marketing Data Engine v2.1 | UI Visibility Patch Applied")