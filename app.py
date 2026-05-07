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

# --- CUSTOM CSS FOR STYLING ---
st.markdown("""
    <style>
    .main { background-color: #f4f7f9; }
    .stMetric {
        background-color: #ffffff;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    hr { margin-top: 1rem; margin-bottom: 1rem; }
    </style>
    """, unsafe_allow_html=True)

# --- LOAD ASSETS ---
@st.cache_resource
def load_all():
    # Ensure this matches your filename on GitHub exactly
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
st.markdown("Comparing **Historical Reality** with **Machine Learning Predictions** to optimize your spend.")
st.divider()

# --- DYNAMIC SIDEBAR FILTERS ---
with st.sidebar:
    st.header("Campaign Parameters")
    st.info("Dropdowns update dynamically based on available data.")
    
    # 1. Company Level
    comp = st.selectbox("Company", sorted(raw_df['Company'].unique()))
    
    # 2. Location Level (Filtered by Company)
    loc_options = sorted(raw_df[raw_df['Company'] == comp]['Location'].unique())
    loc = st.selectbox("Geographic Location", loc_options)
    
    # 3. Channel Level (Filtered by Company + Location)
    chan_options = sorted(raw_df[(raw_df['Company'] == comp) & (raw_df['Location'] == loc)]['Channel_Used'].unique())
    chan = st.selectbox("Marketing Channel", chan_options)
    
    st.divider()
    
    # 4. Audience & Strategy
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
    # Prepare input for XGBoost
    input_row = pd.DataFrame([{
        'Company': comp, 'Campaign_Type': ctype, 'Target_Audience': aud,
        'Duration': 30, 'Channel_Used': chan, 'Location': loc,
        'Language': lang, 'Customer_Segment': seg
    }])
    
    # Apply Encoding
    try:
        for col, le in encoders.items():
            input_row[col] = le.transform(input_row[col])
        
        # Predict: Expected shape [ConvRate, AcqCost, ROI]
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
            st.warning("⚠️ No exact historical match for this specific combination.")
            st.caption("Try changing the filters in the sidebar.")

    with col2:
        st.subheader("🤖 AI Prediction (XGBoost)")
        # ml_pred indices: 0=ConvRate, 1=AcqCost, 2=ROI
        roi_delta = ml_pred[2] - lookup['ROI'].mean()
        
        st.metric("Predicted ROI", f"{ml_pred[2]:.2f}x", delta=f"{roi_delta:.2f} vs Market Avg")
        st.metric("Predicted Conv. Rate", f"{ml_pred[0]*100:.2f}%")
        st.metric("Predicted Acq. Cost", f"${ml_pred[1]:,.2f}")

    # --- STRATEGIC INSIGHT ---
    st.divider()
    with st.expander("💡 Strategic Analysis"):
        if ml_pred[2] > 3.0:
            st.success("**High Performance Expected:** This campaign configuration shows strong potential for scaling.")
        elif ml_pred[2] < 1.5:
            st.error("**Caution:** Predicted ROI is low. Consider changing the Channel or Target Audience.")
        else:
            st.info("**Steady Performance:** This setup is expected to yield average market returns.")

else:
    st.info("👈 Select your campaign parameters on the left and click 'Run Intelligence Report' to start.")

# --- FOOTER ---
st.markdown("---")
st.caption("Marketing Data Engine v2.0 | Powered by XGBoost & Historical Analytics")