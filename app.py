# import streamlit as st
# import librosa
# import numpy as np
# import tensorflow as tf
# import matplotlib.pyplot as plt
# import librosa.display

# # --- 1. CONFIGURATION ---
# # Ensure 'final_heartbeat_model.h5' is in the same folder as this script
# MODEL_PATH = 'final_heartbeat_model.h5' 
# CLASSES = ['normal', 'murmur', 'extrastole', 'extrahls', 'artifact']

# # --- 2. MODEL LOADING ---
# @st.cache_resource
# def load_heart_model():
#     # Loading the legacy .h5 format
#     return tf.keras.models.load_model(MODEL_PATH)

# # --- 3. PREPROCESSING (Exact match to Training) ---
# def preprocess_audio(audio_path, sr=22050, duration=10, n_mfcc=52):
#     """
#     Standardizes input audio to match the training data format:
#     - Load at 22050Hz for 10 seconds
#     - Pad if shorter than 10s
#     - Extract 52 MFCCs and take the mean across time
#     - Reshape to (1, 52, 1) for the model input
#     """
#     # 1. Load audio
#     X, _ = librosa.load(audio_path, sr=sr, duration=duration)
    
#     # 2. Fix length (Padding)
#     input_length = sr * duration
#     if len(X) < input_length:
#         X = librosa.util.fix_length(X, size=input_length)
    
#     # 3. Extract MFCCs (Mean across time)
#     # Using .T and axis=0 ensures we get 52 features representing the whole clip
#     mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sr, n_mfcc=n_mfcc).T, axis=0)
    
#     # 4. Reshape for LSTM/Conv1D: (batch, features, 1)
#     mfccs = mfccs.reshape(1, n_mfcc, 1)
#     return mfccs, X

# # --- 4. STREAMLIT UI DESIGN ---
# st.set_page_config(page_title="Heart Sound Classifier", page_icon="❤️", layout="centered")

# st.title("❤️ Heartbeat Sound Classification")
# st.markdown("""
# Upload a heartbeat audio file (**WAV format**) to analyze cardiac sounds. 
# The system detects patterns like **Murmurs**, **Extrastoles**, and **Normal** beats using deep learning.
# """)

# uploaded_file = st.file_uploader("📤 Upload Heartbeat Audio", type=["wav"])

# if uploaded_file is not None:
#     # Play Audio
#     st.divider()
#     st.audio(uploaded_file, format='audio/wav')
    
#     with st.spinner('🔍 Analyzing heartbeat patterns...'):
#         try:
#             # Load Model
#             model = load_heart_model()
            
#             # Preprocess
#             features, raw_audio = preprocess_audio(uploaded_file)
            
#             # Predict
#             prediction = model.predict(features, verbose=0)
            
#             # Get Top Result
#             top_idx = np.argmax(prediction)
#             predicted_class = CLASSES[top_idx]
#             confidence = prediction[0][top_idx] * 100
            
#             # --- 5. RESULTS DISPLAY ---
#             st.success(f"### Prediction: **{predicted_class.upper()}**")
#             st.metric(label="Confidence Score", value=f"{confidence:.2f}%")
            
#             # --- 6. VISUAL PROBABILITY BREAKDOWN (Sorted) ---
#             st.write("### 📊 Diagnosis Probability")
            
#             # Pair classes with probabilities and sort (Descending)
#             class_probs = sorted(zip(CLASSES, prediction[0]), key=lambda x: x[1], reverse=True)
            
#             # Display sorted list with progress bars
#             for cls, prob in class_probs:
#                 col1, col2, col3 = st.columns([2, 5, 1])
#                 with col1:
#                     st.write(f"**{cls.capitalize()}**")
#                 with col2:
#                     st.progress(float(prob))
#                 with col3:
#                     st.write(f"{prob*100:.1f}%")
            
#             st.divider()

#             # --- 7. AUDIO VISUALIZATION ---
#             st.write("### 📈 Signal Waveform")
#             fig, ax = plt.subplots(figsize=(10, 3))
#             librosa.display.waveshow(raw_audio, sr=22050, ax=ax, color='#ff4b4b')
#             ax.set_title("Time-Domain Analysis")
#             ax.set_xlabel("Time (seconds)")
#             ax.set_ylabel("Amplitude")
#             st.pyplot(fig)

#         except Exception as e:
#             st.error(f"Error processing file: {e}")

# else:
#     st.info("Please upload a .wav file to begin analysis.")

# # --- FOOTER ---
# st.markdown("---")
# st.caption("⚠️ **Disclaimer:** This tool is for educational purposes only and should not be used as a replacement for professional medical diagnosis.")















import streamlit as st
import librosa
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import librosa.display

# --- 1. CONFIGURATION ---
MODEL_PATH = 'final_heartbeat_model.h5' 
CLASSES = ['normal', 'murmur', 'extrastole', 'extrahls', 'artifact']

# --- 2. RECOMMENDATION DATA ---
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

# --- 3. MODEL LOADING ---
@st.cache_resource
def load_heart_model():
    return tf.keras.models.load_model(MODEL_PATH)

# --- 4. PREPROCESSING ---
def preprocess_audio(audio_path, sr=22050, duration=10, n_mfcc=52):
    X, _ = librosa.load(audio_path, sr=sr, duration=duration)
    input_length = sr * duration
    if len(X) < input_length:
        X = librosa.util.fix_length(X, size=input_length)
    mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sr, n_mfcc=n_mfcc).T, axis=0)
    mfccs = mfccs.reshape(1, n_mfcc, 1)
    return mfccs, X

# --- 5. STREAMLIT UI DESIGN ---
st.set_page_config(page_title="CardiaSense AI", page_icon="❤️", layout="centered")

st.title("❤️ Heartbeat Sound Classification")
st.markdown("""
Upload a heartbeat audio file (**WAV format**) to analyze cardiac sounds. 
The system detects patterns like **Murmurs**, **Extrastoles**, and **Normal** beats using deep learning.
""")

uploaded_file = st.file_uploader("📤 Upload Heartbeat Audio", type=["wav"])

if uploaded_file is not None:
    st.divider()
    st.audio(uploaded_file, format='audio/wav')
    
    with st.spinner('🔍 Analyzing heartbeat patterns...'):
        try:
            model = load_heart_model()
            features, raw_audio = preprocess_audio(uploaded_file)
            prediction = model.predict(features, verbose=0)
            
            top_idx = np.argmax(prediction)
            predicted_class = CLASSES[top_idx]
            confidence = prediction[0][top_idx] * 100
            
            # --- RESULTS DISPLAY ---
            st.success(f"### Prediction: **{predicted_class.upper()}**")
            st.metric(label="Confidence Score", value=f"{confidence:.2f}%")
            
            # --- PROBABILITY BREAKDOWN ---
            st.write("### 📊 Diagnosis Probability")
            class_probs = sorted(zip(CLASSES, prediction[0]), key=lambda x: x[1], reverse=True)
            
            for cls, prob in class_probs:
                col1, col2, col3 = st.columns([2, 5, 1])
                with col1:
                    st.write(f"**{cls.capitalize()}**")
                with col2:
                    st.progress(float(prob))
                with col3:
                    st.write(f"{prob*100:.1f}%")
            
            st.divider()

            # --- RECOMMENDATIONS SECTION (The New Professional UI) ---
            rec = RECOMMENDATIONS.get(predicted_class)
            st.subheader(f"{rec['icon']} Clinical Insights: {predicted_class.capitalize()}")
            
            # Create two columns for Description and Steps
            info_col, step_col = st.columns([1, 1])
            
            with info_col:
                st.info(f"**Description:**\n\n{rec['info']}")
                
            with step_col:
                st.warning("**Recommended Next Steps:**")
                for step in rec['steps']:
                    st.write(f"- {step}")

            st.divider()

            # --- AUDIO VISUALIZATION ---
            st.write("### 📈 Signal Waveform")
            fig, ax = plt.subplots(figsize=(10, 3))
            librosa.display.waveshow(raw_audio, sr=22050, ax=ax, color='#ff4b4b')
            ax.set_title("Time-Domain Analysis (Phonocardiogram)")
            ax.set_xlabel("Time (seconds)")
            ax.set_ylabel("Amplitude")
            st.pyplot(fig)

        except Exception as e:
            st.error(f"Error processing file: {e}")

else:
    st.info("Please upload a .wav file to begin analysis.")

# --- FOOTER ---
st.markdown("---")
st.caption("⚠️ **Disclaimer:** This tool is for educational purposes only. It uses AI to identify patterns and does not provide a confirmed medical diagnosis. Always consult a certified cardiologist.")