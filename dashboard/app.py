import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

from scipy.stats import skew, kurtosis
from scipy.fft import fft, fftfreq

# ===============================
# PAGE CONFIG
# ===============================

st.set_page_config(
    page_title="AI-Based RFI Detection",
    layout="wide"
)

# ===============================
# FEATURE EXTRACTION
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
""")

st.divider()

# ===============================
# FILE UPLOAD
# ===============================

st.header("📂 Upload RF Signal Data")

uploaded_file = st.file_uploader(
    "Upload RF signal file (CSV format)",
    type=["csv"]
)

st.info("""
**Expected input**
- CSV file
- Numeric signal values
- Minimum 100 samples
""")

# ===============================
# MAIN LOGIC
# ===============================

if uploaded_file is not None:
    try:
        data = pd.read_csv(uploaded_file)

        st.subheader("📄 Uploaded Data Preview")
        st.dataframe(data.head())

        # Validation
        if not np.issubdtype(data.dtypes[0], np.number):
            st.error("❌ Invalid file: Data must be numeric.")
            st.stop()

        if data.shape[0] < 100:
            st.error("❌ Invalid file: Signal too short.")
            st.stop()

        st.success("✅ File validated successfully.")

        # Prepare signal
        signal = data.values.flatten()
        features = np.array(extract_features(signal)).reshape(1, -1)

        # ===============================
        # RFI DETECTION
        # ===============================

        st.divider()
        st.header("🚨 RFI Detection Output")

        detector = joblib.load("models/rfi_detector.pkl")
        detection = detector.predict(features)[0]

        if detection == 1:
            st.error("⚠️ Interference Detected: YES")
        else:
            st.success("✅ Interference Detected: NO")

        st.markdown("""
        **Explanation:**  
        The model checks statistical patterns of the signal.  
        Abnormal variations indicate interference.
        """)

        # ===============================
        # SIGNAL VISUALIZATION
        # ===============================

        st.divider()
        st.header("📈 Signal Visualization")

        fs = 1000
        time = np.arange(len(signal)) / fs

        # Time-domain
        fig, ax = plt.subplots()
        ax.plot(time, signal)
        ax.set_title("Time-Domain Signal")
        ax.set_xlabel("Time (seconds)")
        ax.set_ylabel("Amplitude")
        st.pyplot(fig)

        st.markdown("Shows signal behavior over time.")

        # FFT
        yf = np.abs(fft(signal))
        xf = fftfreq(len(signal), 1 / fs)

        fig, ax = plt.subplots()
        ax.plot(xf[:len(xf)//2], yf[:len(yf)//2])
        ax.set_title("Frequency Spectrum (FFT)")
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Magnitude")
        st.pyplot(fig)

        st.markdown("Highlights interference frequencies.")

        # Spectrogram
        fig, ax = plt.subplots()
        ax.specgram(signal, Fs=fs)
        ax.set_title("Spectrogram (Time–Frequency)")
        ax.set_xlabel("Time")
        ax.set_ylabel("Frequency")
        st.pyplot(fig)

        st.markdown("Shows when interference occurs.")

        # ===============================
        # RFI CLASSIFICATION
        # ===============================

        st.divider()
        st.header("🧠 RFI Classification Output")

        classifier = joblib.load("models/rfi_classifier.pkl")
        class_pred = classifier.predict(features)[0]

        label_map = {
            1: "Narrowband Interference",
            2: "Broadband Interference",
            3: "Impulsive Interference"
        }

        st.success(f"Detected Type: {label_map[class_pred]}")

        st.markdown("""
        **Why this matters:**  
        Different interference types require different mitigation methods.
        """)

    except Exception as e:
        st.error("❌ Error processing file.")
        st.stop()

else:
    st.success("Dashboard ready. Please upload a signal file.")
