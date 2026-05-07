import streamlit as st
import joblib
import pandas as pd

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="Marketing Analytics Dashboard",
    page_icon="📈",
    layout="wide"
)

# --- CUSTOM STYLING ---
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .stMetric {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .sidebar .sidebar-content {
        background-color: #ffffff;
    }
    </style>
    """, unsafe_allow_html=True)

# --- LOAD ASSETS ---
@st.cache_resource
def load_assets():
    try:
        data = joblib.load('marketing_lookup_model.pkl')
        return data
    except FileNotFoundError:
        st.error("⚠️ Model file 'marketing_lookup_model.pkl' not found. Please ensure it is in the same folder as this script.")
        return None

data = load_assets()

if data:
    lookup = data['lookup_table']
    global_stats = data['global_stats']
    options = data['options']

    # --- SIDEBAR / INPUT SECTION ---
    st.title("📈 Marketing Campaign Performance Analyzer")
    st.markdown("Select campaign parameters to view historical performance metrics based on 200,000 records.")
    st.divider()

    with st.container():
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("🏢 Corporate Context")
            company = st.selectbox("Company", options['Company'])
            location = st.selectbox("Geographic Location", options['Location'])
            language = st.selectbox("Target Language", options['Language'])

        with col2:
            st.subheader("🎯 Strategy")
            campaign_type = st.selectbox("Campaign Type", options['Campaign_Type'])
            target_audience = st.selectbox("Target Audience", options['Target_Audience'])
            segment = st.selectbox("Customer Segment", options['Customer_Segment'])

        with col3:
            st.subheader("📺 Distribution")
            channel = st.selectbox("Marketing Channel", options['Channel_Used'])
            st.info("The duration is currently calculated as a historical average for these categories.")

    st.markdown("###") # Vertical spacer
    
    # --- ACTION BUTTON ---
    if st.button("🚀 Analyze Performance", type="primary", use_container_width=True):
        
        # Filtering logic
        match = lookup[
            (lookup['Company'] == company) &
            (lookup['Campaign_Type'] == campaign_type) &
            (lookup['Target_Audience'] == target_audience) &
            (lookup['Channel_Used'] == channel) &
            (lookup['Location'] == location) &
            (lookup['Language'] == language) &
            (lookup['Customer_Segment'] == segment)
        ]

        st.markdown("---")
        
        if not match.empty:
            res = match.iloc[0]
            
            # Display Metrics
            m1, m2, m3 = st.columns(3)
            
            with m1:
                st.metric(
                    label="Avg. Conversion Rate", 
                    value=f"{res['Conversion_Rate']*100:.2f}%",
                    help="The percentage of visitors who completed the desired action."
                )
            with m2:
                st.metric(
                    label="Avg. Acquisition Cost", 
                    value=f"${res['Acquisition_Cost']:,.2f}",
                    help="The average cost to acquire one customer in this segment."
                )
            with m3:
                st.metric(
                    label="Average ROI", 
                    value=f"{res['ROI']:.2f}x",
                    help="Return on Investment (Multiple of spend)."
                )
            
            st.success("✅ Results generated from exact historical matches.")
            
        else:
            # Fallback to Global Stats
            st.warning("📡 No exact historical match found for this unique combination. Showing Global Averages as a baseline:")
            
            m1, m2, m3 = st.columns(3)
            m1.metric("Global Conversion", f"{global_stats['Conversion_Rate']*100:.2f}%")
            m2.metric("Global Avg. Cost", f"${global_stats['Acquisition_Cost']:,.2f}")
            m3.metric("Global Avg. ROI", f"{global_stats['ROI']:.2f}x")

# --- FOOTER ---
st.markdown("---")
st.caption("Marketing Data Engine v1.0 | Built with Python & Streamlit")