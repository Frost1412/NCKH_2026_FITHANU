"""
Standalone evaluation script for Speech Emotion Recognition (SER).

Features:
- Works with local fine-tuned Wav2Vec2 model folder (e.g. ./final_model)
- Parses RAVDESS filenames directly (no dependency on utils.py)
- Evaluates on either test split or full dataset
- Exports research-friendly artifacts:
  * metrics.csv
  * classification_report.csv
  * confusion_matrix.csv
  * confusion_matrix.png
  * evaluation_results.json
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import librosa
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2ForSequenceClassification

import matplotlib.pyplot as plt
import seaborn as sns


EMOTION_CODE_TO_LABEL: Dict[str, str] = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate SER model for research reporting")
    parser.add_argument("--dataset_dir", type=str, default="dataset", help="Path to RAVDESS dataset")
    parser.add_argument("--model_dir", type=str, default="final_model", help="Path to trained model directory")
    parser.add_argument("--output_dir", type=str, default="evaluation_results", help="Directory to save outputs")
    parser.add_argument("--split", type=str, choices=["test", "all"], default="test", help="Evaluate on test split or all data")
    parser.add_argument("--test_size", type=float, default=0.1, help="Test ratio when split=test")
    parser.add_argument("--val_size", type=float, default=0.1, help="Validation ratio when split=test")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--sample_rate", type=int, default=16000, help="Target sample rate")
    parser.add_argument("--max_audio_len", type=float, default=4.0, help="Maximum audio length in seconds")
    parser.add_argument("--batch_size", type=int, default=8, help="Inference batch size")
    return parser.parse_args()


def parse_ravdess_label(file_path: str) -> str:
    """Parse emotion label from RAVDESS filename pattern.

    Expected pattern example: 03-01-05-01-02-01-12.wav
    Emotion code is the 3rd block.
    """
    name = Path(file_path).stem
    parts = name.split("-")
    if len(parts) < 3:
        raise ValueError(f"Invalid RAVDESS filename: {file_path}")
    emotion_code = parts[2]
    if emotion_code not in EMOTION_CODE_TO_LABEL:
        raise ValueError(f"Unknown emotion code '{emotion_code}' in file: {file_path}")
    return EMOTION_CODE_TO_LABEL[emotion_code]


def build_dataframe(dataset_dir: str) -> pd.DataFrame:
    wav_paths = sorted(str(p) for p in Path(dataset_dir).rglob("*.wav"))
    if not wav_paths:
        raise FileNotFoundError(f"No .wav files found in dataset_dir={dataset_dir}")

    rows = []
    for path in wav_paths:
        try:
            label = parse_ravdess_label(path)
            rows.append({"path": path, "label": label})
        except ValueError:
            # Skip files not matching RAVDESS naming convention
            continue

    if not rows:
        raise RuntimeError("No valid RAVDESS files were parsed. Please check dataset structure and filenames.")

    df = pd.DataFrame(rows)
    return df


def find_candidate_dataset_dirs(search_roots: List[Path], max_candidates: int = 10) -> List[Tuple[str, int]]:
    """Find directories containing wav files to help users set --dataset_dir correctly."""
    candidates: List[Tuple[str, int]] = []
    seen = set()

    for root in search_roots:
        if not root.exists() or not root.is_dir():
            continue

        for wav_file in root.rglob("*.wav"):
            parent = str(wav_file.parent.resolve())
            if parent in seen:
                continue
            seen.add(parent)

            wav_count = len(list(Path(parent).glob("*.wav")))
            candidates.append((parent, wav_count))
            if len(candidates) >= max_candidates:
                return candidates

    return candidates


def train_val_test_split(
    df: pd.DataFrame,
    test_size: float,
    val_size: float,
    seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create stratified train/val/test split.

    val_size is interpreted as ratio over full dataset.
    """
    if test_size <= 0 or val_size <= 0 or test_size + val_size >= 1:
        raise ValueError("Require 0 < test_size, val_size and (test_size + val_size) < 1")

    train_val_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=seed,
        stratify=df["label"],
    )

    val_ratio_in_train_val = val_size / (1.0 - test_size)
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=val_ratio_in_train_val,
        random_state=seed,
        stratify=train_val_df["label"],
    )
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)


def load_audio(path: str, sample_rate: int, max_audio_len: float) -> np.ndarray:
    audio, _ = librosa.load(path, sr=sample_rate, mono=True)
    max_len = int(sample_rate * max_audio_len)
    if len(audio) > max_len:
        audio = audio[:max_len]
    return audio


def batched_predict(
    model: Wav2Vec2ForSequenceClassification,
    extractor: Wav2Vec2FeatureExtractor,
    audios: List[np.ndarray],
    device: str,
    sample_rate: int,
    max_audio_len: float,
    batch_size: int,
) -> np.ndarray:
    max_length = int(sample_rate * max_audio_len)
    all_preds: List[np.ndarray] = []

    model.eval()
    with torch.no_grad():
        for i in range(0, len(audios), batch_size):
            batch = audios[i : i + batch_size]
            inputs = extractor(
                batch,
                sampling_rate=sample_rate,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length,
            )
            logits = model(inputs["input_values"].to(device)).logits
            preds = logits.argmax(dim=-1).cpu().numpy()
            all_preds.append(preds)

    return np.concatenate(all_preds, axis=0)


def save_outputs(
    output_dir: str,
    labels: List[str],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metrics: Dict[str, float],
) -> None:
    os.makedirs(output_dir, exist_ok=True)

    # metrics.csv
    metrics_df = pd.DataFrame(
        {
            "Metric": ["Accuracy", "Balanced Accuracy", "F1 Macro", "F1 Weighted"],
            "Value": [
                metrics["accuracy"],
                metrics["balanced_accuracy"],
                metrics["f1_macro"],
                metrics["f1_weighted"],
            ],
        }
    )
    metrics_df.to_csv(os.path.join(output_dir, "metrics.csv"), index=False)

    # classification_report.csv
    report_dict = classification_report(
        y_true,
        y_pred,
        target_names=labels,
        output_dict=True,
        zero_division=0,
    )
    report_df = pd.DataFrame(report_dict).transpose()
    report_df.to_csv(os.path.join(output_dir, "classification_report.csv"), index=True)

    # confusion_matrix.csv
    cm = confusion_matrix(y_true, y_pred)
    cm_df = pd.DataFrame(cm, index=labels, columns=labels)
    cm_df.to_csv(os.path.join(output_dir, "confusion_matrix.csv"), index=True)

    # confusion_matrix.png
    cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-12)
    plt.figure(figsize=(9, 7))
    sns.heatmap(
        cm_norm,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        cbar_kws={"label": "Proportion"},
    )
    plt.title("Normalized Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "confusion_matrix.png"), dpi=300)
    plt.close()

    # evaluation_results.json
    payload = {
        "metrics": metrics,
        "num_samples": int(len(y_true)),
        "labels": labels,
        "artifacts": {
            "metrics_csv": "metrics.csv",
            "classification_report_csv": "classification_report.csv",
            "confusion_matrix_csv": "confusion_matrix.csv",
            "confusion_matrix_png": "confusion_matrix.png",
        },
    }
    with open(os.path.join(output_dir, "evaluation_results.json"), "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def main() -> None:
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 70)
    print("SER MODEL EVALUATION")
    print("=" * 70)
    print(f"Model dir:   {args.model_dir}")
    print(f"Dataset dir: {args.dataset_dir}")
    print(f"Output dir:  {args.output_dir}")
    print(f"Device:      {device}")
    print("=" * 70)

    # Load model + extractor
    print("\n[1/4] Loading model and feature extractor...")
    model = Wav2Vec2ForSequenceClassification.from_pretrained(args.model_dir, local_files_only=True)
    extractor = Wav2Vec2FeatureExtractor.from_pretrained(args.model_dir, local_files_only=True)
    model.to(device).eval()

    # Build dataframe and labels
    print("[2/4] Building evaluation dataframe...")
    try:
        df = build_dataframe(args.dataset_dir)
    except FileNotFoundError as e:
        cwd = Path.cwd()
        search_roots = [cwd, cwd.parent, Path(args.dataset_dir)]
        candidates = find_candidate_dataset_dirs(search_roots)

        print("\n❌ DATASET NOT FOUND")
        print(str(e))
        print("\nGợi ý:")
        print("1) Đảm bảo bạn đã tải/giải nén RAVDESS (.wav) vào máy.")
        print("2) Chạy lại với đường dẫn dataset đúng, ví dụ:")
        print("   python evaluate.py --dataset_dir \"D:/RAVDESS\"")

        if candidates:
            print("\n📁 Các thư mục có thể dùng làm --dataset_dir:")
            for path, count in candidates:
                print(f"   - {path}  ({count} wav files trực tiếp)")
        else:
            print("\nKhông tìm thấy file .wav nào quanh thư mục dự án hiện tại.")

        return
    labels = sorted(df["label"].unique().tolist())
    label2id = {label: idx for idx, label in enumerate(labels)}

    if args.split == "test":
        try:
            _, _, eval_df = train_val_test_split(df, args.test_size, args.val_size, args.seed)
        except ValueError as e:
            print(f"⚠️ Stratified split failed ({e}). Falling back to full dataset evaluation.")
            eval_df = df.copy()
    else:
        eval_df = df.copy()

    print(f"   Evaluating {len(eval_df)} samples across {len(labels)} classes")

    # Load audio + inference
    print("[3/4] Running inference...")
    audios: List[np.ndarray] = []
    y_true: List[int] = []
    for _, row in eval_df.iterrows():
        audio = load_audio(row["path"], args.sample_rate, args.max_audio_len)
        audios.append(audio)
        y_true.append(label2id[row["label"]])

    y_true_np = np.array(y_true)
    y_pred_np = batched_predict(
        model=model,
        extractor=extractor,
        audios=audios,
        device=device,
        sample_rate=args.sample_rate,
        max_audio_len=args.max_audio_len,
        batch_size=args.batch_size,
    )

    # Compute metrics
    print("[4/4] Computing metrics and saving outputs...")
    metrics = {
        "accuracy": float(accuracy_score(y_true_np, y_pred_np)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true_np, y_pred_np)),
        "f1_macro": float(f1_score(y_true_np, y_pred_np, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(y_true_np, y_pred_np, average="weighted", zero_division=0)),
    }

    save_outputs(args.output_dir, labels, y_true_np, y_pred_np, metrics)

    print("\n✅ Evaluation complete")
    print(f"   Accuracy:          {metrics['accuracy'] * 100:.2f}%")
    print(f"   Balanced Accuracy: {metrics['balanced_accuracy'] * 100:.2f}%")
    print(f"   F1 Macro:          {metrics['f1_macro'] * 100:.2f}%")
    print(f"   F1 Weighted:       {metrics['f1_weighted'] * 100:.2f}%")
    print(f"\n📁 Artifacts saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
