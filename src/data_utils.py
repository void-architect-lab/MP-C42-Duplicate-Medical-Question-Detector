"""Dataset download, cleaning, tokenization, and pair DataLoader helpers."""
from __future__ import annotations

import os
import re
import urllib.request
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from torch.utils.data import Dataset

DATA_URL = "https://raw.githubusercontent.com/curai/medical-question-pair-dataset/master/mqp.csv"
PAD_TOKEN_ID = 50256
SEED = 42


def download_dataset(path: str | Path = "data/mqp.csv") -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size > 1000:
        return path
    print(f"Downloading MQP dataset from: {DATA_URL}")
    request = urllib.request.Request(DATA_URL, headers={"User-Agent": "MP-C42-student-project/1.0"})
    try:
        with urllib.request.urlopen(request, timeout=45) as response, path.open("wb") as out:
            out.write(response.read())
    except Exception as exc:
        if path.exists() and path.stat().st_size < 1000:
            path.unlink(missing_ok=True)
        raise RuntimeError(
            "Could not download the public MQP CSV. Check your internet connection, then retry. "
            f"Source: {DATA_URL}. Original error: {exc}"
        ) from exc
    return path


def _find_column(columns: list[str], candidates: tuple[str, ...]) -> str:
    normalized = {re.sub(r"[^a-z0-9]", "", str(c).lower()): str(c) for c in columns}
    for candidate in candidates:
        key = re.sub(r"[^a-z0-9]", "", candidate.lower())
        if key in normalized:
            return normalized[key]
    raise ValueError(f"Could not identify a required column among {columns}. Expected one of {candidates}.")


def _label_to_int(value: Any) -> int:
    if isinstance(value, bool):
        return int(value)
    text = str(value).strip().lower()
    if text in {"1", "1.0", "similar", "same", "duplicate", "true", "yes"}:
        return 1
    if text in {"0", "0.0", "dissimilar", "different", "different question", "false", "no"}:
        return 0
    try:
        numeric = int(float(text))
        if numeric in (0, 1):
            return numeric
    except ValueError:
        pass
    raise ValueError(f"Unrecognized MQP label: {value!r}; expected 0/1 or similar/dissimilar.")


def read_mqp(path: str | Path) -> pd.DataFrame:
    """Read Curai's headerless `dr_id, question_1, question_2, label` CSV.

    The first field is the doctor ID (1-11), not a row ID. The public file has
    no header row, so reading it with pandas' default header handling silently
    loses the first example and misidentifies all four columns.
    """
    raw = pd.read_csv(path, header=None, dtype="string", keep_default_na=False)
    if raw.shape[1] < 4:
        # Friendly support for a manually downloaded/renamed CSV with headers.
        headed = pd.read_csv(path, dtype="string", keep_default_na=False)
        q1_col = _find_column(list(headed.columns), ("question_1", "question1", "q1", "question_a", "text_a"))
        q2_col = _find_column(list(headed.columns), ("question_2", "question2", "q2", "question_b", "text_b"))
        label_col = _find_column(list(headed.columns), ("label", "similarity", "is_similar", "duplicate", "class"))
        df = pd.DataFrame({"question_1": headed[q1_col], "question_2": headed[q2_col], "label": headed[label_col]})
    else:
        # Curai format: doctor ID, question 1, question 2, binary label.
        df = pd.DataFrame({"question_1": raw.iloc[:, -3], "question_2": raw.iloc[:, -2], "label": raw.iloc[:, -1]})

    df["question_1"] = df["question_1"].astype("string").str.strip()
    df["question_2"] = df["question_2"].astype("string").str.strip()
    df["label"] = df["label"].map(_label_to_int)
    df = df.dropna(subset=["question_1", "question_2", "label"])
    df = df[(df.question_1.str.len() > 0) & (df.question_2.str.len() > 0)]
    df = df.drop_duplicates(subset=["question_1", "question_2", "label"]).reset_index(drop=True)
    if not set(df.label.unique()).issubset({0, 1}) or df.label.nunique() < 2:
        raise ValueError("Dataset must contain both binary labels: 0 (different) and 1 (similar).")
    return df


def make_pair_tokens(question_1: str, question_2: str, tokenizer: Any, max_length: int = 96) -> list[int]:
    """Encode both questions, reserving room for both sides to avoid truncating Q2 entirely."""
    prefix = tokenizer.encode("Question 1: ")
    separator = tokenizer.encode("\nQuestion 2: ")
    budget = max(2, max_length - len(prefix) - len(separator))
    q1_budget = max(1, budget // 2)
    q2_budget = max(1, budget - q1_budget)
    q1_ids = tokenizer.encode(str(question_1), disallowed_special=())[:q1_budget]
    q2_ids = tokenizer.encode(str(question_2), disallowed_special=())[:q2_budget]
    ids = prefix + q1_ids + separator + q2_ids
    return ids[:max_length] if ids else [PAD_TOKEN_ID]


class MedicalPairDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, tokenizer: Any, max_length: int = 96) -> None:
        self.items: list[tuple[list[int], int]] = []
        for row in frame.itertuples(index=False):
            ids = make_pair_tokens(row.question_1, row.question_2, tokenizer, max_length=max_length)
            self.items.append((ids, int(row.label)))

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        token_ids, label = self.items[index]
        return torch.tensor(token_ids, dtype=torch.long), label


def pair_collate(batch: list[tuple[torch.Tensor, int]]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    sequences, labels = zip(*batch)
    lengths = torch.tensor([len(seq) for seq in sequences], dtype=torch.long)
    max_len = int(lengths.max().item())
    inputs = torch.full((len(sequences), max_len), PAD_TOKEN_ID, dtype=torch.long)
    for i, seq in enumerate(sequences):
        inputs[i, :len(seq)] = seq
    return inputs, torch.tensor(labels, dtype=torch.long), lengths
