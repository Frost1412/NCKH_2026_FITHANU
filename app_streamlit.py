import streamlit as st
import os
import io
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px  # ĐÃ THÊM ĐỂ FIX LỖI 'px' not defined
from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2FeatureExtractor
import torch
import soundfile as sf
from io import BytesIO
from pathlib import Path
import librosa

EMOTION_CODE_TO_LABEL = {
    '01': 'neutral',
    '02': 'calm',
    '03': 'happy',
    '04': 'sad',
    '05': 'angry',
    '06': 'fearful',
    '07': 'disgust',
    '08': 'surprised',
}


def load_wav(file_path, target_sr=16000):
    audio, sr = librosa.load(file_path, sr=target_sr, mono=True)
    return audio, sr


def build_dataframe(dataset_dir):
    rows = []
    for wav_path in Path(dataset_dir).rglob('*.wav'):
        parts = wav_path.stem.split('-')
        if len(parts) < 3:
            continue
        label = EMOTION_CODE_TO_LABEL.get(parts[2])
        if label is None:
            continue
        rows.append({'path': str(wav_path), 'label': label})
    return pd.DataFrame(rows)

st.set_page_config(page_title='SER Classroom Dashboard', layout='wide')

st.title('SER Classroom Dashboard (Wav2Vec2 Fine-tuned with Long-Audio Support)')
st.markdown("""
Upload speech audio (English). The app loads your fine-tuned model from `./final_model`.  
For long audio (>10s), it uses **overlapped chunking (4s chunk, 3s overlap)** to analyze the entire recording and displays emotion probability trends over time (8 separate subplots, like research figures).
**Emotions**: angry, calm, disgust, fearful, happy, neutral, sad, surprised.
""")

# ================= CONFIG =================
MODEL_DIR = os.path.join(os.path.dirname(__file__), 'final_model')
LOG_PATH = 'logs/emotion_log.csv'
os.makedirs('logs', exist_ok=True)

CHUNK_SEC = 4
OVERLAP_SEC = 1  # Optimal như trong paper (stride = 1s)
MIN_DURATION_FOR_CHUNKING = 10  # giây

EMOTIONS = ['angry', 'calm', 'disgust', 'fearful', 'happy', 'neutral', 'sad', 'surprised']

@st.cache_resource
def load_model():
    if os.path.isdir(MODEL_DIR):
        try:
            model = Wav2Vec2ForSequenceClassification.from_pretrained(MODEL_DIR)
            feat = Wav2Vec2FeatureExtractor.from_pretrained("facebook/wav2vec2-base")
            st.success("✅ Fine-tuned model loaded from final_model!")
        except Exception as e:
            st.error(f"Error loading custom model: {e}")
            st.info("Falling back to pretrained model...")
            model_id = 'superb/wav2vec2-base-superb-er'
            model = Wav2Vec2ForSequenceClassification.from_pretrained(model_id)
            feat = Wav2Vec2FeatureExtractor.from_pretrained(model_id)
    else:
        st.warning('No custom model found. Using pretrained superb/wav2vec2-base-superb-er')
        model_id = 'superb/wav2vec2-base-superb-er'
        model = Wav2Vec2ForSequenceClassification.from_pretrained(model_id)
        feat = Wav2Vec2FeatureExtractor.from_pretrained(model_id)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    model.eval()
    return model, feat, device

model, feat, device = load_model()

# ================= SAMPLE METADATA =================
if not os.path.exists('metadata.csv'):
    sample_meta = pd.DataFrame({
        'student_id': ['S001', 'S002', 'S003'],
        'name': ['Alice Johnson', 'Bob Smith', 'Carol Davis']
    })
    sample_meta.to_csv('metadata.csv', index=False)
    st.info("Generated sample metadata.csv. Edit it for real students.")

meta = pd.read_csv('metadata.csv')

# ================= SIDEBAR =================
st.sidebar.header('Session Controls')
student = st.sidebar.selectbox('Select student', options=meta['student_id'].tolist())
note = st.sidebar.text_input('Optional note (e.g., activity)')

# ================= MAIN LAYOUT =================
col1, col2 = st.columns([1, 1])

with col1:
    st.header('Live Inference')
    uploaded = st.file_uploader('Upload WAV file (speech)', type=['wav'])

    if uploaded is not None:
        audio_bytes = uploaded.read()
        tmp = 'tmp_stream.wav'
        with open(tmp, 'wb') as f:
            f.write(audio_bytes)

        audio, sr = load_wav(tmp, target_sr=feat.sampling_rate)
        duration = len(audio) / sr

        st.audio(audio_bytes)
        st.write(f"Audio duration: {duration:.2f} seconds")

        if duration <= MIN_DURATION_FOR_CHUNKING:
            # Audio ngắn: predict trực tiếp
            max_length = feat.sampling_rate * CHUNK_SEC
            audio = audio[:max_length]
            inputs = feat(audio, sampling_rate=feat.sampling_rate, return_tensors='pt', padding=True)
            input_values = inputs['input_values'].to(device)
            with torch.no_grad():
                out = model(input_values)
                probs = torch.softmax(out.logits, dim=-1).cpu().numpy()[0]

            df_prob = pd.DataFrame({'Emotion': EMOTIONS, 'Probability': probs})
            st.subheader('Prediction Probabilities')
            st.table(df_prob.sort_values('Probability', ascending=False).reset_index(drop=True))

            fig = px.bar(df_prob, x='Emotion', y='Probability', title='Emotion Probabilities')
            fig.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig, use_container_width=True)

            top_idx = int(np.argmax(probs))
            top_emotion = EMOTIONS[top_idx]
            top_prob = probs[top_idx]
            st.success(f"**Top prediction**: {top_emotion} ({top_prob:.2%})")

            # Log
            ts = pd.Timestamp.now().isoformat()
            row = pd.DataFrame([{
                'timestamp': ts,
                'student_id': student,
                'emotion': top_emotion,
                'probability': float(top_prob),
                'note': note
            }])
            if os.path.exists(LOG_PATH):
                logs = pd.read_csv(LOG_PATH)
                logs = pd.concat([logs, row], ignore_index=True)
            else:
                logs = row
            logs.to_csv(LOG_PATH, index=False)
            st.info('Logged to logs/emotion_log.csv')

        else:
            # Audio dài: tách chunk + overlap + vẽ timeline giống ảnh bạn gửi
            st.info(f"Long audio detected ({duration:.2f}s). Analyzing entire recording with overlapped chunking (4s chunk, 3s overlap)...")

            chunk_len = feat.sampling_rate * CHUNK_SEC
            stride = feat.sampling_rate * (CHUNK_SEC - OVERLAP_SEC)

            probs_over_time = []
            chunk_times = []

            for start in range(0, len(audio) - chunk_len + 1, stride):
                chunk = audio[start:start + chunk_len]
                inputs = feat(chunk, sampling_rate=feat.sampling_rate, return_tensors='pt', padding=True)
                input_values = inputs['input_values'].to(device)
                with torch.no_grad():
                    out = model(input_values)
                    probs = torch.softmax(out.logits, dim=-1).cpu().numpy()[0]
                probs_over_time.append(probs)
                chunk_times.append(start / feat.sampling_rate)

            probs_over_time = np.array(probs_over_time)

            # Aggregate total probability
            total_probs = np.mean(probs_over_time, axis=0)
            df_prob = pd.DataFrame({'Emotion': EMOTIONS, 'Probability': total_probs})
            st.subheader('Aggregated Prediction Probabilities (Entire Audio)')
            st.table(df_prob.sort_values('Probability', ascending=False).reset_index(drop=True))

            top_idx = int(np.argmax(total_probs))
            top_emotion = EMOTIONS[top_idx]
            top_prob = total_probs[top_idx]
            st.success(f"**Overall top prediction**: {top_emotion} ({top_prob:.2%})")

            # Vẽ timeline probability trends - giống ảnh bạn gửi (8 subplot riêng)
            st.subheader('Emotion Probability Trends over Time')
            fig, axs = plt.subplots(4, 2, figsize=(16, 12), sharex=True, sharey=True)
            fig.suptitle(f'Emotion Probability Trends over Time - {os.path.basename(uploaded.name)}', fontsize=18, y=1.02)

            for i, emotion in enumerate(EMOTIONS):
                row = i // 2
                col = i % 2
                axs[row, col].plot(chunk_times, probs_over_time[:, i], color='blue', linewidth=1.2, alpha=0.9)
                axs[row, col].set_title(emotion.capitalize(), fontsize=14)
                axs[row, col].set_ylabel('Probability', fontsize=12)
                axs[row, col].set_ylim(0, 1.05)
                axs[row, col].grid(True, alpha=0.3, linestyle='--')
                axs[row, col].tick_params(labelsize=10)

            # X-label chung cho hàng dưới
            for col in range(2):
                axs[3, col].set_xlabel('Time (seconds)', fontsize=12)

            plt.tight_layout(rect=[0, 0, 1, 0.96])  # Để title không bị cắt
            st.pyplot(fig)

            # Log aggregated result
            ts = pd.Timestamp.now().isoformat()
            row = pd.DataFrame([{
                'timestamp': ts,
                'student_id': student,
                'emotion': top_emotion,
                'probability': float(top_prob),
                'note': note + f" (long audio: {duration:.2f}s)"
            }])
            if os.path.exists(LOG_PATH):
                logs = pd.read_csv(LOG_PATH)
                logs = pd.concat([logs, row], ignore_index=True)
            else:
                logs = row
            logs.to_csv(LOG_PATH, index=False)
            st.info('Logged aggregated result to logs/emotion_log.csv')

with col2:
    st.header('Class Statistics & Timeline')
    if os.path.exists(LOG_PATH):
        logs = pd.read_csv(LOG_PATH)
        logs['timestamp'] = pd.to_datetime(logs['timestamp'])

        st.subheader('Overall Emotion Distribution')
        counts = logs['emotion'].value_counts().reset_index()
        counts.columns = ['Emotion', 'Count']
        fig_pie = px.pie(counts, names='Emotion', values='Count', title='Emotion Distribution (All Sessions)')
        st.plotly_chart(fig_pie, use_container_width=True)

        st.subheader('Emotion Timeline per Student')
        sel_student = st.selectbox('Choose student', options=logs['student_id'].unique())
        student_logs = logs[logs['student_id'] == sel_student].copy()
        student_logs = student_logs.sort_values('timestamp')
        fig_line = px.line(student_logs, x='timestamp', y='probability', color='emotion',
                           markers=True, title=f'Timeline for {sel_student}')
        st.plotly_chart(fig_line, use_container_width=True)

        st.subheader('Recent Logs (newest first)')
        st.dataframe(logs.sort_values('timestamp', ascending=False).reset_index(drop=True))
    else:
        st.info('No logs yet. Process some audio to generate logs.')

st.markdown("---")
st.markdown("**Notes**: Long audio (>10s) is fully analyzed with overlapped chunking (4s chunk, 3s overlap) for complete emotion trends. Short audio uses single 4s segment for realtime performance.")