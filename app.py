import streamlit as st
import joblib
import pandas as pd
import numpy as np

st.set_page_config(page_title="AI Marketing Intelligence", layout="wide")

@st.cache_resource
def load_all():
    return joblib.load('marketing_hybrid_model.pkl')

assets = load_all()
lookup = assets['lookup_table']
ml_model = assets['ml_model']
encoders = assets['label_encoders']
raw_df = assets['df']

st.title("🎯 Hybrid Marketing Forecaster")
st.markdown("This tool combines **Historical Averages** with **Machine Learning Predictions**.")

# --- DYNAMIC UI LOGIC ---
with st.sidebar:
    st.header("Campaign Filters")
    
    # 1. Company
    comp = st.selectbox("Company", sorted(raw_df['Company'].unique()))
    
    # 2. Filter locations based on Company
    loc_options = sorted(raw_df[raw_df['Company'] == comp]['Location'].unique())
    loc = st.selectbox("Location", loc_options)
    
    # 3. Filter Channels based on Company + Location
    chan_options = sorted(raw_df[(raw_df['Company'] == comp) & (raw_df['Location'] == loc)]['Channel_Used'].unique())
    chan = st.selectbox("Channel", chan_options)
    
    # 4. Filter others (Remaining inputs)
    aud = st.selectbox("Target Audience", sorted(raw_df['Target_Audience'].unique()))
    ctype = st.selectbox("Campaign Type", sorted(raw_df['Campaign_Type'].unique()))
    lang = st.selectbox("Language", sorted(raw_df['Language'].unique()))
    seg = st.selectbox("Customer Segment", sorted(raw_df['Customer_Segment'].unique()))

# --- CALCULATION ---
if st.button("Generate Intelligence Report", type="primary"):
    
    # 1. Look for Historical Match
    match = lookup[
        (lookup['Company'] == comp) & (lookup['Location'] == loc) & 
        (lookup['Channel_Used'] == chan) & (lookup['Target_Audience'] == aud) &
        (lookup['Campaign_Type'] == ctype) & (lookup['Language'] == lang) &
        (lookup['Customer_Segment'] == seg)
    ]

    # 2. Generate ML Prediction (Always)
    input_data = pd.DataFrame([{
        'Company': comp, 'Campaign_Type': ctype, 'Target_Audience': aud,
        'Duration': 30, 'Channel_Used': chan, 'Location': loc,
        'Language': lang, 'Customer_Segment': seg
    }])
    
    # Encode for ML
    for col, le in encoders.items():
        input_data[col] = le.transform(input_data[col])
    
    ml_pred = ml_model.predict(input_data)[0]

    # --- DISPLAY ---
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📊 Historical Data (Actuals)")
        if not match.empty:
            res = match.iloc[0]
            st.metric("Avg ROI", f"{res['ROI']:.2f}x")
            st.metric("Avg Conv. Rate", f"{res['Conversion_Rate']*100:.2f}%")
        else:
            st.warning("No exact historical match found for this specific path.")

    with col2:
        st.subheader("🤖 AI Prediction (XGBoost)")
        st.metric("Predicted ROI", f"{ml_pred[2]:.2f}x", delta=f"{(ml_pred[2]-lookup['ROI'].mean()):.2f} vs Avg")
        st.metric("Predicted Conv. Rate", f"{ml_pred[0]*100:.2f}%")

    st.success("Analysis Complete. The AI model accounts for patterns even where historical data is missing.")