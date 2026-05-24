# NCKH_2026_FITHANU
# Speech Emotion Recognition (SER) - Wav2Vec2 Fine-tuned

Dự án này huấn luyện và đánh giá mô hình **Speech Emotion Recognition** dựa trên **Wav2Vec2** cho bài toán nhận diện cảm xúc từ giọng nói.  
Project hiện có:

- Script huấn luyện: `train_model_FIXED.py`
- Script đánh giá nghiên cứu: `file_full_optimized.py`
- Script đánh giá gọn: `evaluate.py`
- Ứng dụng demo tương tác: `app_streamlit.py`
- Mô hình đã fine-tune: `final_model/`

---

## Dataset

Dự án sử dụng bộ dữ liệu **RAVDESS Emotional Speech Audio**.

- Link tải trên Kaggle:  
  https://www.kaggle.com/datasets/uwrfkaggler/ravdess-emotional-speech-audio
- Mô tả nhanh:
  - 1,440 file `.wav`
  - 24 diễn viên chuyên nghiệp
  - 8 nhãn cảm xúc: `angry`, `calm`, `disgust`, `fearful`, `happy`, `neutral`, `sad`, `surprised`
  - File audio-only, 48 kHz

> Lưu ý: nếu bạn muốn chạy lại huấn luyện/đánh giá, hãy giải nén dataset vào thư mục `dataset/` trong project hoặc chỉ rõ đường dẫn dataset khi chạy script.

---

## Tính năng chính

- Fine-tune Wav2Vec2 cho phân loại cảm xúc
- Đánh giá test set với các metric:
  - Accuracy
  - Balanced Accuracy / WA
  - Macro F1
  - Weighted F1
- Xuất biểu đồ và bảng cho nghiên cứu
- Demo Streamlit để upload file `.wav` và xem dự đoán cảm xúc
- Hỗ trợ audio dài bằng **chunking + overlap**

---

## Cấu trúc thư mục

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

## Yêu cầu môi trường

Khuyến nghị dùng Python 3.10+ và tạo virtual environment riêng.

### Tạo và kích hoạt `venv` trên PowerShell

```powershell
Set-Location "C:\Users\ACER\OneDrive - hanu.edu.vn\Desktop\ser_models\ser_optimized_v2"
python -m venv venv
Set-ExecutionPolicy -Scope Process Bypass -Force
.\venv\Scripts\Activate.ps1
```

### Cài thư viện cần thiết

```powershell
pip install streamlit torch transformers librosa pandas numpy matplotlib plotly scikit-learn soundfile datasets evaluate seaborn tqdm
```

---

## Chạy đánh giá mô hình

### 1) Script đánh giá gọn

```powershell
Set-Location "C:\Users\ACER\OneDrive - hanu.edu.vn\Desktop\ser_models\ser_optimized_v2"
python evaluate.py
```

Kết quả sẽ được lưu trong thư mục `evaluation_results/`.

### 2) Script đánh giá nghiên cứu đầy đủ

```powershell
Set-Location "C:\Users\ACER\OneDrive - hanu.edu.vn\Desktop\ser_models\ser_optimized_v2"
python file_full_optimized.py
```

Script này sẽ sinh thêm nhiều bảng và hình minh họa cho phần viết paper, lưu trong `final_results/`.

---

## Chạy ứng dụng Streamlit

```powershell
Set-Location "C:\Users\ACER\OneDrive - hanu.edu.vn\Desktop\ser_models\ser_optimized_v2"
streamlit run app_streamlit.py
```

Ứng dụng sẽ mở tại:

- Local URL: `http://localhost:8501`

Tính năng của app:

- Upload file `.wav`
- Dự đoán cảm xúc theo thời gian thực
- Hỗ trợ audio dài bằng chunking + overlap
- Lưu log kết quả vào `logs/emotion_log.csv`

---

## Đầu ra quan trọng

- `test_metrics.csv` / `evaluation_results/metrics.csv`  
  Thống kê metric đánh giá
- `evaluation_results/confusion_matrix.png`  
  Ma trận nhầm lẫn
- `final_results/`  
  Bộ file nghiên cứu đầy đủ: SOTA comparison, ablation study, noise robustness, inference time, learning curve, ...

---

## Ghi chú về dữ liệu và nhãn

Dự án hiện parse nhãn từ tên file RAVDESS theo quy ước:

- `01 = neutral`
- `02 = calm`
- `03 = happy`
- `04 = sad`
- `05 = angry`
- `06 = fearful`
- `07 = disgust`
- `08 = surprised`

Ví dụ:

- `03-01-06-01-02-01-12.wav` → `fearful`

---

## Trích dẫn dataset RAVDESS

Nếu dùng dataset này trong nghiên cứu, hãy trích dẫn:

**Livingstone SR, Russo FA (2018)**. *The Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS): A dynamic, multimodal set of facial and vocal expressions in North American English.* PLoS ONE 13(5): e0196391.  
DOI: https://doi.org/10.1371/journal.pone.0196391

---

## License dataset

Theo trang Kaggle, dataset này được phát hành dưới giấy phép:

- **CC BY-NC-SA 4.0**

---

## Tác giả / ghi chú

Project này được tối ưu cho mục tiêu:

- nghiên cứu Speech Emotion Recognition
- đánh giá mô hình bằng metric chuẩn
- tạo biểu đồ và bảng phục vụ báo cáo / paper

*Dự án này được xây dựng bởi đội nghiên cứu khoa học số 10, khoa Công nghệ thông tin, trường Đại học Hà Nội. Mục tiêu là để phục vụ cho việc nghiên cứu và báo cáo khoa học*