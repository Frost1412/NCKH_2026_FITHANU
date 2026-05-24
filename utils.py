"""Shared helpers for SER scripts and Streamlit app."""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import librosa
import pandas as pd
from sklearn.model_selection import train_test_split

EMOTION_CODE_TO_LABEL = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised",
}


def parse_ravdess_label(file_path: str) -> str:
    name = Path(file_path).stem
    parts = name.split("-")
    if len(parts) < 3:
        raise ValueError(f"Invalid RAVDESS filename: {file_path}")
    code = parts[2]
    if code not in EMOTION_CODE_TO_LABEL:
        raise ValueError(f"Unknown emotion code {code} in file {file_path}")
    return EMOTION_CODE_TO_LABEL[code]


def build_dataframe(dataset_dir: str) -> pd.DataFrame:
    rows = []
    for wav_path in Path(dataset_dir).rglob("*.wav"):
        try:
            label = parse_ravdess_label(str(wav_path))
        except Exception:
            continue
        rows.append({"path": str(wav_path), "label": label})
    return pd.DataFrame(rows)


def train_val_test_split(
    df: pd.DataFrame,
    test_size: float = 0.1,
    val_size: float = 0.1,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if df.empty:
        raise ValueError("DataFrame is empty")

    try:
        train_val_df, test_df = train_test_split(
            df,
            test_size=test_size,
            random_state=seed,
            stratify=df["label"],
        )
        val_ratio = val_size / (1.0 - test_size)
        train_df, val_df = train_test_split(
            train_val_df,
            test_size=val_ratio,
            random_state=seed,
            stratify=train_val_df["label"],
        )
    except Exception:
        train_val_df, test_df = train_test_split(df, test_size=test_size, random_state=seed)
        val_ratio = val_size / (1.0 - test_size)
        train_df, val_df = train_test_split(train_val_df, test_size=val_ratio, random_state=seed)

    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)


def load_wav(file_path: str, target_sr: int = 16000):
    audio, sr = librosa.load(file_path, sr=target_sr, mono=True)
    return audio, sr
