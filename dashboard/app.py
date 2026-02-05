import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

from scipy.stats import skew, kurtosis

# ===============================
# PAGE CONFIG
# ===============================

st.set_page_config(
    page_title="AI-Based RFI Detection",
    layout="wide"
)

# ===============================
# FEATURE EXTRACTION FUNCTION
# ===============================

def extract_features(signal):
    return [
        np.mean(signal),
        np.std(signal),
        np.max(signal),
        np.min(signal),
        skew(signal),
        kurtosis(signal)
    ]

# ===============================
# HEADER
# ===============================

st.title("📡 AI-Based Radio Frequency Interference Detection & Classification")

st.markdown("""
This dashboard allows users to upload radio frequency signal data and automatically:
- Detect whether interference is present
- Identify the type of interference
- Visualize signals in time and frequency domains

The system uses Machine Learning models trained on RF signal features.
""")

st.divider()

# ===============================
# FILE UPLOAD SECTION
# ===============================

st.header("📂 Upload RF Signal Data")

uploaded_file = st.file_uploader(
    "Upload RF signal file (CSV format)",
    type=["csv"]
)

st.info("""
**Expected input:**
- A CSV file containing numeric RF signal values
- Minimum 100 samples required
""")

# ===============================
# FILE VALIDATION
# ===============================

if uploaded_file is not None:
    try:
        data = pd.read_csv(uploaded_file)

        st.subheader("📄 Uploaded Data Preview")
        st.dataframe(data.head())

        # Numeric validation
        if not np.issubdtype(data.dtypes[0], np.number):
            st.error("❌ Invalid file: Data must contain only numeric values.")
            st.stop()

        # Length validation
        if data.shape[0] < 100:
            st.error("❌ Invalid file: Signal data is too short for analysis.")
            st.stop()

        st.success("✅ File validated successfully. Ready for analysis.")

    except Exception:
        st.error("❌ Error reading file. Please upload a valid CSV file.")
        st.stop()

    # ===============================
    # RFI DETECTION
    # ===============================

    st.divider()
    st.header("🚨 RFI Detection Output")

    signal_values = data.values.flatten()
    features = np.array(extract_features(signal_values)).reshape(1, -1)

    detector = joblib.load("models/rfi_detector.pkl")
    detection_result = detector.predict(features)[0]

    if detection_result == 1:
        st.error("⚠️ Interference Detected: YES")
    else:
        st.success("✅ Interference Detected: NO")

    st.markdown("""
    ### 🧠 What does this result mean?

    - **YES** → The signal shows abnormal behavior compared to clean RF signals  
    - **NO** → The signal follows normal RF characteristics  

    ### 🔍 How was this decided?

    The system extracts statistical features such as:
    - Average signal level
    - Signal variation
    - Peak values
    - Shape characteristics (skewness & kurtosis)

    These features are analyzed using a **Random Forest Machine Learning model**
    trained on known RF interference patterns.
    """)

    st.divider()

    # ===============================
    # PLACEHOLDERS (NEXT STEPS)
    # ===============================

    st.header("📈 Signal Visualization")
    st.write("⏳ Time-domain, FFT, and spectrogram plots will appear here.")

    st.divider()

    st.header("🧠 RFI Classification Output")
    st.write("➡️ Interference type will be shown here in the next step.")

    st.divider()

    st.header("📊 Model Performance Metrics")
    st.write("➡️ Accuracy, Precision, Recall, F1-score will be shown here.")

else:
    st.success("Dashboard loaded successfully. Awaiting file upload.")
