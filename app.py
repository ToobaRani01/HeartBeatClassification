
# import streamlit as st
# import librosa
# import numpy as np
# import tensorflow as tf
# import matplotlib.pyplot as plt
# import librosa.display

# # --- 1. CONFIGURATION ---
# MODEL_PATH = 'final_heartbeat_model.h5' 
# CLASSES = ['normal', 'murmur', 'extrastole', 'extrahls', 'artifact']

# # --- 2. RECOMMENDATION DATA ---
# RECOMMENDATIONS = {
#     'normal': {
#         'icon': "✅",
#         'info': "The heartbeat rhythm is within normal parameters. No significant murmurs or structural anomalies detected.",
#         'steps': ["Maintain regular cardio exercise.", "Schedule annual routine check-ups.", "Maintain a balanced, low-sodium diet."]
#     },
#     'murmur': {
#         'icon': "🔊",
#         'info': "Abnormal 'whooshing' sounds detected. This often indicates turbulent blood flow across heart valves.",
#         'steps': ["Consult a cardiologist for an Echocardiogram (Echo).", "Monitor for shortness of breath during exercise.", "Avoid heavy stimulants like high-dosage caffeine."]
#     },
#     'extrastole': {
#         'icon': "💓",
#         'info': "Irregular beats detected that disrupt the normal 'lub-dub' sequence (skipped or extra beats).",
#         'steps': ["Track frequency of palpitations.", "Check electrolyte levels (Potassium/Magnesium).", "Consult a doctor for a 24-hour Holter monitor test."]
#     },
#     'extrahls': {
#         'icon': "⚠️",
#         'info': "Additional heart sounds detected (S3 or S4 gallops). This may indicate changes in heart chamber stiffness.",
#         'steps': ["Professional clinical auscultation is required.", "Discuss blood pressure management with a physician.", "Consider a cardiac stress test if symptoms persist."]
#     },
#     'artifact': {
#         'icon': "🚫",
#         'info': "The recording contains significant background noise or poor sensor contact.",
#         'steps': ["Ensure a quiet environment for recording.", "Keep the stethoscope/sensor still against the skin.", "Retry the recording with deeper, steady breaths."]
#     }
# }

# # --- 3. MODEL LOADING ---
# @st.cache_resource
# def load_heart_model():
#     return tf.keras.models.load_model(MODEL_PATH)

# # --- 4. PREPROCESSING ---
# def preprocess_audio(audio_path, sr=22050, duration=10, n_mfcc=52):
#     X, _ = librosa.load(audio_path, sr=sr, duration=duration)
#     input_length = sr * duration
#     if len(X) < input_length:
#         X = librosa.util.fix_length(X, size=input_length)
#     mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sr, n_mfcc=n_mfcc).T, axis=0)
#     mfccs = mfccs.reshape(1, n_mfcc, 1)
#     return mfccs, X

# # --- 5. STREAMLIT UI DESIGN ---
# st.set_page_config(page_title="CardiaSense AI", page_icon="❤️", layout="centered")

# st.title("❤️ Heartbeat Sound Classification")
# st.markdown("""
# Upload a heartbeat audio file (**WAV format**) to analyze cardiac sounds. 
# The system detects patterns like **Murmurs**, **Extrastoles**, and **Normal** beats using deep learning.
# """)

# uploaded_file = st.file_uploader("📤 Upload Heartbeat Audio", type=["wav"])

# if uploaded_file is not None:
#     st.divider()
#     st.audio(uploaded_file, format='audio/wav')
    
#     with st.spinner('🔍 Analyzing heartbeat patterns...'):
#         try:
#             model = load_heart_model()
#             features, raw_audio = preprocess_audio(uploaded_file)
#             prediction = model.predict(features, verbose=0)
            
#             top_idx = np.argmax(prediction)
#             predicted_class = CLASSES[top_idx]
#             confidence = prediction[0][top_idx] * 100
            
#             # --- RESULTS DISPLAY ---
#             st.success(f"### Prediction: **{predicted_class.upper()}**")
#             st.metric(label="Confidence Score", value=f"{confidence:.2f}%")
            
#             # --- PROBABILITY BREAKDOWN ---
#             st.write("### 📊 Diagnosis Probability")
#             class_probs = sorted(zip(CLASSES, prediction[0]), key=lambda x: x[1], reverse=True)
            
#             for cls, prob in class_probs:
#                 col1, col2, col3 = st.columns([2, 5, 1])
#                 with col1:
#                     st.write(f"**{cls.capitalize()}**")
#                 with col2:
#                     st.progress(float(prob))
#                 with col3:
#                     st.write(f"{prob*100:.1f}%")
            
#             st.divider()

#             # --- RECOMMENDATIONS SECTION (The New Professional UI) ---
#             rec = RECOMMENDATIONS.get(predicted_class)
#             st.subheader(f"{rec['icon']} Clinical Insights: {predicted_class.capitalize()}")
            
#             # Create two columns for Description and Steps
#             info_col, step_col = st.columns([1, 1])
            
#             with info_col:
#                 st.info(f"**Description:**\n\n{rec['info']}")
                
#             with step_col:
#                 st.warning("**Recommended Next Steps:**")
#                 for step in rec['steps']:
#                     st.write(f"- {step}")

#             st.divider()

#             # --- AUDIO VISUALIZATION ---
#             st.write("### 📈 Signal Waveform")
#             fig, ax = plt.subplots(figsize=(10, 3))
#             librosa.display.waveshow(raw_audio, sr=22050, ax=ax, color='#ff4b4b')
#             ax.set_title("Time-Domain Analysis (Phonocardiogram)")
#             ax.set_xlabel("Time (seconds)")
#             ax.set_ylabel("Amplitude")
#             st.pyplot(fig)

#         except Exception as e:
#             st.error(f"Error processing file: {e}")

# else:
#     st.info("Please upload a .wav file to begin analysis.")

# # --- FOOTER ---
# st.markdown("---")
# st.caption("⚠️ **Disclaimer:** This tool is for educational purposes only. It uses AI to identify patterns and does not provide a confirmed medical diagnosis. Always consult a certified cardiologist.")











import streamlit as st
import librosa
import librosa.display
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter
import soundfile as sf
import io

# --- 1. CONFIGURATION ---
MODEL_PATH = 'final_heartbeat_model.h5' 
CLASSES = ['normal', 'murmur', 'extrastole', 'extrahls', 'artifact']

# --- 2. NOISE REMOVAL UTILITIES ---
def butter_bandpass(lowcut, highcut, fs, order=3):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def apply_bandpass_filter(data, lowcut=20.0, highcut=500.0, fs=22050):
    data = np.nan_to_num(data)
    b, a = butter_bandpass(lowcut, highcut, fs)
    y = lfilter(b, a, data)
    y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
    return y

# --- 3. RECOMMENDATION DATA ---
RECOMMENDATIONS = {
    'normal': {
        'icon': "✅",
        'info': "The heartbeat rhythm is within normal parameters. No significant murmurs or structural anomalies detected.",
        'steps': ["Maintain regular cardio exercise.", "Schedule annual routine check-ups.", "Maintain a balanced, low-sodium diet."]
    },
    'murmur': {
        'icon': "🔊",
        'info': "Abnormal 'whooshing' sounds detected. This often indicates turbulent blood flow across heart valves.",
        'steps': ["Consult a cardiologist for an Echocardiogram (Echo).", "Monitor for shortness of breath during exercise.", "Avoid heavy stimulants like high-dosage caffeine."]
    },
    'extrastole': {
        'icon': "💓",
        'info': "Irregular beats detected that disrupt the normal 'lub-dub' sequence (skipped or extra beats).",
        'steps': ["Track frequency of palpitations.", "Check electrolyte levels (Potassium/Magnesium).", "Consult a doctor for a 24-hour Holter monitor test."]
    },
    'extrahls': {
        'icon': "⚠️",
        'info': "Additional heart sounds detected (S3 or S4 gallops). This may indicate changes in heart chamber stiffness.",
        'steps': ["Professional clinical auscultation is required.", "Discuss blood pressure management with a physician.", "Consider a cardiac stress test if symptoms persist."]
    },
    'artifact': {
        'icon': "🚫",
        'info': "The recording contains significant background noise or poor sensor contact.",
        'steps': ["Ensure a quiet environment for recording.", "Keep the stethoscope/sensor still against the skin.", "Retry the recording with deeper, steady breaths."]
    }
}

# --- 4. MODEL LOADING ---
@st.cache_resource
def load_heart_model():
    return tf.keras.models.load_model(MODEL_PATH)

# --- 5. PREPROCESSING PIPELINE ---
def process_audio_pipeline(uploaded_file, sr=22050, duration=10, n_mfcc=52):
    X_raw, _ = librosa.load(uploaded_file, sr=sr, duration=duration)
    X_cleaned = apply_bandpass_filter(X_raw, fs=sr)
    
    # Normalization for stability
    if np.max(np.abs(X_cleaned)) > 0:
        X_cleaned = X_cleaned / np.max(np.abs(X_cleaned))
    
    input_length = sr * duration
    if len(X_cleaned) < input_length:
        X_cleaned = librosa.util.fix_length(X_cleaned, size=input_length)
    
    mfccs = np.mean(librosa.feature.mfcc(y=X_cleaned, sr=sr, n_mfcc=n_mfcc).T, axis=0)
    mfccs = mfccs.reshape(1, n_mfcc, 1)
    
    return mfccs, X_raw, X_cleaned

# --- 6. STREAMLIT UI ---
st.set_page_config(page_title="CardiaSense AI", page_icon="❤️", layout="wide")

st.title("❤️ CardiaSense AI: Heartbeat Sound Classification")
st.markdown("Upload a **WAV file**. We apply a **Bandpass Filter (20Hz-500Hz)** to clean the signal before AI diagnosis.")

uploaded_file = st.file_uploader("📤 Upload Heartbeat Audio", type=["wav"])

if uploaded_file is not None:
    with st.spinner('🔍 Analyzing cleaned heartbeat patterns...'):
        try:
            model = load_heart_model()
            features, raw_audio, cleaned_audio = process_audio_pipeline(uploaded_file)
            
            prediction = model.predict(features, verbose=0)
            top_idx = np.argmax(prediction)
            predicted_class = CLASSES[top_idx]
            confidence = prediction[0][top_idx] * 100

            # --- 1. AUDIO PLAYBACK COMPARISON ---
            st.divider()
            col_a, col_b = st.columns(2)
            with col_a:
                st.write("### 🔈 Original Audio")
                st.audio(uploaded_file)
            with col_b:
                st.write("### 🪄 Cleaned Audio (Used for AI)")
                buf = io.BytesIO()
                sf.write(buf, cleaned_audio, 22050, format='wav')
                st.audio(buf)

            # --- 2. PREDICTION & PROBABILITY ---
            st.divider()
            res_col, prob_col = st.columns([1, 1])
            
            with res_col:
                st.success(f"## Prediction: **{predicted_class.upper()}**")
                st.metric(label="Primary Confidence", value=f"{confidence:.2f}%")
                
                # Visual Signal Waveforms
                fig, ax = plt.subplots(2, 1, figsize=(10, 5), sharex=True)
                librosa.display.waveshow(raw_audio, sr=22050, ax=ax[0], color='gray', alpha=0.5)
                ax[0].set_title("Original Waveform (Raw)")
                librosa.display.waveshow(cleaned_audio, sr=22050, ax=ax[1], color='#ff4b4b')
                ax[1].set_title("Filtered Waveform (20Hz - 500Hz)")
                plt.tight_layout()
                st.pyplot(fig)

            with prob_col:
                st.write("### 📊 Diagnosis Probability")
                class_probs = sorted(zip(CLASSES, prediction[0]), key=lambda x: x[1], reverse=True)
                for cls, prob in class_probs:
                    c1, c2, c3 = st.columns([2, 5, 1])
                    with c1: st.write(f"**{cls.capitalize()}**")
                    with c2: st.progress(float(prob))
                    with c3: st.write(f"{prob*100:.1f}%")

            # --- 3. CLINICAL RECOMMENDATIONS ---
            st.divider()
            rec = RECOMMENDATIONS.get(predicted_class)
            st.subheader(f"{rec['icon']} Clinical Insights: {predicted_class.capitalize()}")
            
            info_col, step_col = st.columns([1, 1])
            with info_col:
                st.info(f"**Description:**\n\n{rec['info']}")
            with step_col:
                st.warning("**Recommended Next Steps:**")
                for step in rec['steps']:
                    st.write(f"📍 {step}")

        except Exception as e:
            st.error(f"Error processing file: {e}")
else:
    st.info("Please upload a .wav file to begin analysis.")

st.markdown("---")
st.caption("⚠️ **Disclaimer:** This tool is for educational purposes only. Always consult a certified cardiologist for medical diagnosis.")
