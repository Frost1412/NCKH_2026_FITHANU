# NCKH_2026_FITHANU
# Speech Emotion Recognition (SER) - Wav2Vec2 Fine-tuned

This project trains and evaluates a **Speech Emotion Recognition** model based on **Wav2Vec2** for the task of recognizing emotions from speech.  
Current project components:

- Training script: `train_model_FIXED.py`
- Comprehensive research evaluation script: `file_full_optimized.py`
- Concise evaluation script: `evaluate.py`
- Interactive demo application: `app_streamlit.py`
- Fine-tuned model: `final_model/`

---

## Dataset

The project uses the **RAVDESS Emotional Speech Audio** dataset.

- Kaggle download link:  
  https://www.kaggle.com/datasets/uwrfkaggler/ravdess-emotional-speech-audio
- Quick description:
  - 1,440 `.wav` files
  - 24 professional actors
  - 8 emotion labels: `angry`, `calm`, `disgust`, `fearful`, `happy`, `neutral`, `sad`, `surprised`
  - Audio-only files, 48 kHz

> Note: If you want to rerun the training/evaluation, please extract the dataset into the `dataset/` directory within the project or specify the dataset path when running the scripts.

---

## Key Features

- Fine-tunes Wav2Vec2 for emotion classification
- Evaluates the test set using the following metrics:
  - Accuracy
  - Balanced Accuracy / WA
  - Macro F1
  - Weighted F1
- Exports charts and tables for research purposes
- Streamlit demo for uploading `.wav` files and viewing emotion predictions
- Supports long audio using **chunking + overlap**

---

## Directory Structure

```text
ser_optimized_v2/
├── app_streamlit.py
├── evaluate.py
├── file_full_optimized.py
├── train_model_FIXED.py
├── utils.py
├── dataset/
├── final_model/
├── final_results/
├── evaluation_results/
├── logs/
├── long_audio/
└── README.md
```

---

## Environmental Requirements

It is recommended to use Python 3.10+ and create a dedicated virtual environment.

### Create and activate `venv` on PowerShell

```powershell
Set-Location "C:\Users\ACER\OneDrive - hanu.edu.vn\Desktop\ser_models\ser_optimized_v2"
python -m venv venv
Set-ExecutionPolicy -Scope Process Bypass -Force
.\venv\Scripts\Activate.ps1
```

### Install required libraries

```powershell
pip install streamlit torch transformers librosa pandas numpy matplotlib plotly scikit-learn soundfile datasets evaluate seaborn tqdm
```

---

## Run Model Evaluation

### 1) Concise evaluation script

```powershell
Set-Location "C:\Users\ACER\OneDrive - hanu.edu.vn\Desktop\ser_models\ser_optimized_v2"
python evaluate.py
```

Results will be saved in the `evaluation_results/` directory.

### 2) Comprehensive research evaluation script

```powershell
Set-Location "C:\Users\ACER\OneDrive - hanu.edu.vn\Desktop\ser_models\ser_optimized_v2"
python file_full_optimized.py
```

This script will generate additional tables and illustrations for writing papers, saved in `final_results/`.

---

## Run Streamlit Application

```powershell
Set-Location "C:\Users\ACER\OneDrive - hanu.edu.vn\Desktop\ser_models\ser_optimized_v2"
streamlit run app_streamlit.py
```

The application will open at:

- Local URL: `http://localhost:8501`

App features:

- Upload `.wav` files
- Real-time emotion prediction
- Support for long audio using chunking + overlap
- Save result logs to `logs/emotion_log.csv`

---

## Important Outputs

- `test_metrics.csv` / `evaluation_results/metrics.csv`  
  Evaluation metrics statistics
- `evaluation_results/confusion_matrix.png`  
  Confusion matrix
- `final_results/`  
  Full set of research files: SOTA comparison, ablation study, noise robustness, inference time, learning curve, etc.

---

## Notes on Data and Labels

The project currently parses labels from RAVDESS file names according to the convention:

- `01 = neutral`
- `02 = calm`
- `03 = happy`
- `04 = sad`
- `05 = angry`
- `06 = fearful`
- `07 = disgust`
- `08 = surprised`

Example:

- `03-01-06-01-02-01-12.wav` → `fearful`

---

## RAVDESS Dataset Citation

If you use this dataset in your research, please cite:

**Livingstone SR, Russo FA (2018)**. *The Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS): A dynamic, multimodal set of facial and vocal expressions in North American English.* PLoS ONE 13(5): e0196391.  
DOI: https://doi.org/10.1371/journal.pone.0196391

---

## Dataset License

According to the Kaggle page, this dataset is released under the license:

- **CC BY-NC-SA 4.0**

---

## Author / Notes

This project is optimized for the following goals:

- Researching Speech Emotion Recognition
- Evaluating models using standard metrics
- Generating charts and tables for reports / papers

*This project was built by the research team No. 10, Faculty of Information Technology, Hanoi University. The goal is to serve the research and scientific reporting*