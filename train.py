"""Validation-selected GPT-2 Small fine-tuning for MP-C42.

The recommended Windows launcher uses all cleaned pairs, a stratified 70/15/15
split, four epochs, best-validation-loss checkpoint selection and a decision
threshold selected on the validation split. The held-out test split is never used
to select the threshold or checkpoint.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tiktoken
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, classification_report
from torch.utils.data import DataLoader

from src.data_utils import DATA_URL, MedicalPairDataset, download_dataset, pair_collate, read_mqp
from src.gpt_model import GPT2_SMALL_CONFIG, GPTModel, freeze_for_classification, load_gpt2_small_weights

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "mqp.csv"
OUT_DIR = ROOT / "outputs"
SEED = 42


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def metric_bundle(y_true: list[int], y_pred: list[int]) -> dict:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
        "classification_report": classification_report(y_true, y_pred, labels=[0, 1], target_names=["different", "similar"], zero_division=0, output_dict=True),
    }


def evaluate(
    model: GPTModel,
    loader: DataLoader,
    device: torch.device,
    positive_threshold: float = 0.5,
) -> tuple[dict, list[dict]]:
    """Evaluate using each pair's final non-padding token and an explicit threshold."""
    model.eval()
    ys: list[int] = []
    preds: list[int] = []
    scores: list[float] = []
    losses: list[float] = []
    errors: list[dict] = []
    loss_fn = torch.nn.CrossEntropyLoss()
    with torch.no_grad():
        for inputs, labels, lengths in loader:
            inputs, labels, lengths = inputs.to(device), labels.to(device), lengths.to(device)
            token_logits = model(inputs)
            rows = torch.arange(inputs.size(0), device=device)
            logits = token_logits[rows, lengths - 1, :]
            loss = loss_fn(logits, labels)
            losses.append(float(loss.item()))
            positive_scores = torch.softmax(logits, dim=-1)[:, 1]
            batch_pred = (positive_scores >= positive_threshold).long()
            ys.extend(labels.cpu().tolist())
            preds.extend(batch_pred.cpu().tolist())
            scores.extend(positive_scores.cpu().tolist())
            for true, pred, input_row, length in zip(labels.cpu().tolist(), batch_pred.cpu().tolist(), inputs.cpu(), lengths.cpu().tolist()):
                if true != pred:
                    errors.append({"true_label": int(true), "predicted_label": int(pred), "token_length": int(length), "input_preview": input_row[:min(length, 24)].tolist()})
    metrics = metric_bundle(ys, preds)
    metrics["loss"] = float(np.mean(losses)) if losses else None
    metrics["n"] = len(ys)
    metrics["_y_true"] = ys
    metrics["_y_pred"] = preds
    metrics["_y_score"] = scores
    return metrics, errors


def select_validation_threshold(y_true: list[int], positive_scores: list[float]) -> tuple[float, dict]:
    """Choose a threshold on validation data only; prefer accuracy, then macro-F1, then 0.5."""
    candidates = np.arange(0.30, 0.701, 0.01)
    best = None
    for threshold in candidates:
        predictions = (np.asarray(positive_scores) >= threshold).astype(int)
        accuracy = accuracy_score(y_true, predictions)
        macro_f1 = f1_score(y_true, predictions, average="macro", zero_division=0)
        # Deterministic tie-break: maximize accuracy, then macro-F1, then stay near 0.5.
        key = (float(accuracy), float(macro_f1), -abs(float(threshold) - 0.5))
        if best is None or key > best[0]:
            best = (key, float(threshold), {"accuracy": float(accuracy), "macro_f1": float(macro_f1)})
    assert best is not None
    return best[1], best[2]


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune the course-style GPTModel for MP-C42")
    parser.add_argument("--max-samples", type=int, default=1024, help="Stratified subset size; use 0 for all clean dataset rows")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=96)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--data", type=str, default=str(DATA_PATH))
    parser.add_argument("--skip-report", action="store_true", help="Do not create/update the report after training")
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.max_length < 16:
        parser.error("epochs and batch size must be >= 1, and max-length must be >= 16")

    set_seed(args.seed)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    data_path = Path(args.data)
    if not data_path.exists():
        download_dataset(data_path)
    full_df = read_mqp(data_path)
    dataset_total_clean = len(full_df)

    if args.max_samples > 0 and len(full_df) > args.max_samples:
        sample_df, _ = train_test_split(full_df, train_size=args.max_samples, random_state=args.seed, stratify=full_df["label"])
        used_df = sample_df.reset_index(drop=True)
    else:
        used_df = full_df.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)

    train_df, temp_df = train_test_split(used_df, test_size=0.30, random_state=args.seed, stratify=used_df["label"])
    val_df, test_df = train_test_split(temp_df, test_size=0.50, random_state=args.seed, stratify=temp_df["label"])
    train_df, val_df, test_df = [x.reset_index(drop=True) for x in (train_df, val_df, test_df)]

    print(f"Dataset rows after cleaning: {dataset_total_clean}")
    print(f"Subset/splits: train={len(train_df)}, validation={len(val_df)}, test={len(test_df)}")
    print(f"Label counts in subset: {used_df.label.value_counts().sort_index().to_dict()} (0=different, 1=similar)")
    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    counts = used_df.label.value_counts().reindex([0, 1], fill_value=0)
    ax.bar(["Different (0)", "Similar (1)"], [int(counts[0]), int(counts[1])])
    ax.set_ylabel("Number of pairs")
    ax.set_title("MQP sampled-subset class distribution")
    for idx, value in enumerate([int(counts[0]), int(counts[1])]):
        ax.text(idx, value, str(value), ha="center", va="bottom")
    fig.tight_layout()
    data_plot = OUT_DIR / "data_distribution.png"
    fig.savefig(data_plot, dpi=160)
    plt.close(fig)

    tokenizer = tiktoken.get_encoding("gpt2")
    train_ds = MedicalPairDataset(train_df, tokenizer, args.max_length)
    val_ds = MedicalPairDataset(val_df, tokenizer, args.max_length)
    test_ds = MedicalPairDataset(test_df, tokenizer, args.max_length)
    pin_memory = torch.cuda.is_available()
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=pair_collate, pin_memory=pin_memory)
    train_eval_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=False, collate_fn=pair_collate, pin_memory=pin_memory)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=pair_collate, pin_memory=pin_memory)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, collate_fn=pair_collate, pin_memory=pin_memory)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    print(f"Compute device: {device}")
    if device.type == "cpu":
        print("NOTE: CPU mode is supported but slower. For a faster run, use the laptop NVIDIA GPU and install a CUDA-enabled PyTorch build.")

    model = GPTModel(GPT2_SMALL_CONFIG, num_classes=2)
    print("Downloading/loading pretrained GPT-2 Small weights (about 500+ MB, one-time cache)...")
    weights_path = load_gpt2_small_weights(model)
    model.to(device)
    trainable_params = freeze_for_classification(model)
    print(f"Trainable parameters: {trainable_params:,} / {sum(p.numel() for p in model.parameters()):,}")

    start = time.time()
    before, _ = evaluate(model, test_loader, device)
    before_y = before.pop("_y_true")
    before_pred = before.pop("_y_pred")
    before.pop("_y_score", None)
    print(f"Before fine-tuning: accuracy={before['accuracy']:.4f}, F1={before['f1']:.4f}")

    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=args.learning_rate, weight_decay=0.1)
    loss_fn = torch.nn.CrossEntropyLoss()
    train_losses: list[float] = []
    val_losses: list[float] = []
    best_val_loss = float("inf")
    best_epoch = 0
    best_state = None

    for epoch in range(args.epochs):
        model.train()
        batch_losses = []
        for step, (inputs, labels, lengths) in enumerate(train_loader, start=1):
            inputs, labels, lengths = inputs.to(device), labels.to(device), lengths.to(device)
            optimizer.zero_grad(set_to_none=True)
            all_logits = model(inputs)
            rows = torch.arange(inputs.size(0), device=device)
            logits = all_logits[rows, lengths - 1, :]
            loss = loss_fn(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], max_norm=1.0)
            optimizer.step()
            batch_losses.append(float(loss.item()))
            if step % 20 == 0 or step == len(train_loader):
                print(f"Epoch {epoch + 1}/{args.epochs} | batch {step}/{len(train_loader)} | loss {loss.item():.4f}")
        train_losses.append(float(np.mean(batch_losses)))
        validation, _ = evaluate(model, val_loader, device)
        val_losses.append(float(validation["loss"]))
        print(f"Epoch {epoch + 1}/{args.epochs}: train_loss={train_losses[-1]:.4f}, val_loss={val_losses[-1]:.4f}, val_accuracy={validation['accuracy']:.4f}, val_F1={validation['f1']:.4f}")
        if val_losses[-1] < best_val_loss:
            best_val_loss = val_losses[-1]
            best_epoch = epoch + 1
            best_state = {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    print(f"Selected epoch {best_epoch} using lowest validation loss ({best_val_loss:.4f}).")

    # Tune the decision threshold on validation predictions only; never use test labels for tuning.
    validation, _ = evaluate(model, val_loader, device, positive_threshold=0.5)
    val_y = validation.pop("_y_true")
    validation.pop("_y_pred", None)
    val_scores = validation.pop("_y_score")
    positive_threshold, threshold_val_metrics = select_validation_threshold(val_y, val_scores)
    print(f"Validation-selected similar threshold: {positive_threshold:.2f} "
          f"(validation accuracy={threshold_val_metrics['accuracy']:.4f}, macro-F1={threshold_val_metrics['macro_f1']:.4f})")

    elapsed = time.time() - start
    after, error_rows = evaluate(model, test_loader, device, positive_threshold=positive_threshold)
    after_y = after.pop("_y_true")
    after_pred = after.pop("_y_pred")
    after_scores = after.pop("_y_score")
    print("\n=== Held-out test results (threshold chosen on validation) ===")
    print(f"Before: accuracy={before['accuracy']:.4f}, F1={before['f1']:.4f}, loss={before['loss']:.4f}")
    print(f"After : accuracy={after['accuracy']:.4f}, F1={after['f1']:.4f}, loss={after['loss']:.4f}")
    print(f"Confusion matrix (rows=true [different, similar], columns=predicted [different, similar]): {after['confusion_matrix']}")

    checkpoint_path = OUT_DIR / "duplicate_detector.pt"
    torch.save({"state_dict": model.cpu().state_dict(), "config": GPT2_SMALL_CONFIG,
                "max_length": args.max_length, "seed": args.seed,
                "positive_threshold": positive_threshold, "best_epoch": best_epoch}, checkpoint_path)
    # Verify the saved checkpoint is loadable and obtain metrics from the reloaded model.
    reloaded = GPTModel(GPT2_SMALL_CONFIG, num_classes=2)
    saved = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    reloaded.load_state_dict(saved["state_dict"])
    reloaded.to(device)
    reloaded_metrics, _ = evaluate(reloaded, test_loader, device, positive_threshold=positive_threshold)
    reloaded_metrics.pop("_y_true", None); reloaded_metrics.pop("_y_pred", None); reloaded_metrics.pop("_y_score", None)
    print(f"Checkpoint saved and reloaded successfully: {checkpoint_path}")

    # Curves and confusion matrix used by the report.
    fig, ax = plt.subplots(figsize=(7, 4))
    epochs_axis = list(range(1, args.epochs + 1))
    ax.plot(epochs_axis, train_losses, marker="o", label="Training loss")
    ax.plot(epochs_axis, val_losses, marker="o", label="Validation loss")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Cross-entropy loss")
    ax.set_title("GPT-2 classifier fine-tuning loss")
    ax.grid(True, alpha=0.25); ax.legend(); fig.tight_layout()
    loss_plot = OUT_DIR / "loss_curve.png"
    fig.savefig(loss_plot, dpi=160); plt.close(fig)

    cm = np.array(after["confusion_matrix"])
    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    im = ax.imshow(cm, interpolation="nearest")
    ax.set_title("Held-out test confusion matrix")
    ax.set_xlabel("Predicted label"); ax.set_ylabel("True label")
    ax.set_xticks([0, 1], labels=["Different", "Similar"])
    ax.set_yticks([0, 1], labels=["Different", "Similar"])
    for (i, j), value in np.ndenumerate(cm):
        ax.text(j, i, str(int(value)), ha="center", va="center")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04); fig.tight_layout()
    cm_plot = OUT_DIR / "confusion_matrix.png"
    fig.savefig(cm_plot, dpi=160); plt.close(fig)

    # Save manually auditable test predictions and examples of model errors.
    predictions_path = OUT_DIR / "test_predictions.csv"
    test_out = test_df.copy()
    test_out["predicted_label"] = after_pred
    test_out["correct"] = (test_out.label.astype(int) == test_out.predicted_label.astype(int))
    test_out.to_csv(predictions_path, index=False)
    wrong = test_out[~test_out.correct].head(3)
    error_examples = []
    for row in wrong.itertuples(index=False):
        error_examples.append({"question_1": row.question_1, "question_2": row.question_2,
                               "expected": int(row.label), "predicted": int(row.predicted_label)})

    # 5 self-written prompts for the UI demo; predictions use the saved, fine-tuned model.
    demo_pairs = [
        ("What are common symptoms of diabetes?", "Which signs may indicate that someone has diabetes?", 1),
        ("Can aspirin cause stomach bleeding?", "How can stomach bleeding be prevented?", 0),
        ("What causes high blood pressure?", "Why might a person develop hypertension?", 1),
        ("Is insulin used to treat type 1 diabetes?", "What is the role of insulin in type 1 diabetes?", 1),
        ("Does a sore throat always mean strep infection?", "What are the symptoms of a urinary tract infection?", 0),
    ]
    reloaded.eval()
    demo_outputs = []
    with torch.no_grad():
        for q1, q2, expected in demo_pairs:
            ids = torch.tensor([__import__("src.data_utils", fromlist=["make_pair_tokens"]).make_pair_tokens(q1, q2, tokenizer, args.max_length)], dtype=torch.long, device=device)
            logits = reloaded(ids)[0, -1, :]
            probs = torch.softmax(logits, dim=-1)
            pred = int(float(probs[1].item()) >= positive_threshold)
            demo_outputs.append({"question_1": q1, "question_2": q2, "expected_label": expected,
                                 "predicted_label": pred, "output": "SIMILAR" if pred == 1 else "DIFFERENT",
                                 "expected_output": "SIMILAR" if expected == 1 else "DIFFERENT",
                                 "correct": bool(pred == expected), "p_similar": float(probs[1].item()), "threshold": positive_threshold})

    results = {
        "project_id": "MP-C42", "project_title": "Duplicate Medical Question Detector",
        "dataset_url": DATA_URL, "dataset_rows_clean": dataset_total_clean,
        "subset_used": len(used_df), "subset_label_counts": {str(int(k)): int(v) for k, v in used_df.label.value_counts().sort_index().items()},
        "dataset_samples": [{"question_1": str(r.question_1), "question_2": str(r.question_2), "label": int(r.label)} for r in full_df.head(2).itertuples(index=False)],
        "data_distribution_path": str(data_plot),
        "split_sizes": {"train": len(train_df), "validation": len(val_df), "test": len(test_df)},
        "seed": args.seed, "max_length": args.max_length, "batch_size": args.batch_size,
        "epochs": args.epochs, "learning_rate": args.learning_rate,
        "best_epoch_by_validation_loss": best_epoch, "positive_threshold_selected_on_validation": positive_threshold,
        "threshold_validation_metrics": threshold_val_metrics,
        "optimizer": "AdamW", "weight_decay": 0.1,
        "base_model": "GPT-2 Small (124M)", "weights_source": "openai-community/gpt2/model.safetensors",
        "weights_path": str(weights_path), "device": str(device),
        "total_parameters": sum(p.numel() for p in reloaded.parameters()), "trainable_parameters": trainable_params,
        "training_seconds": elapsed,
        "before": before, "after": after, "reloaded_test_metrics": reloaded_metrics,
        "train_losses": train_losses, "validation_losses": val_losses,
        "loss_curve_path": str(loss_plot), "confusion_matrix_path": str(cm_plot),
        "test_predictions_path": str(predictions_path), "error_examples": error_examples,
        "demo_pairs": demo_outputs,
        "limitations": ["not a medical diagnostic tool", "English-only, small subset fine-tuning", "dataset licence/terms should be verified before redistribution"],
    }
    with (OUT_DIR / "results.json").open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    (OUT_DIR / "classification_report.txt").write_text(classification_report(after_y, after_pred, labels=[0, 1], target_names=["different", "similar"], zero_division=0), encoding="utf-8")
    print(f"Saved measured results to {OUT_DIR / 'results.json'}")

    if not args.skip_report:
        try:
            from build_report import build_report
            build_report(ROOT, require_results=True)
        except Exception as exc:
            print(f"Training succeeded, but automatic report generation needs attention: {exc}", file=sys.stderr)


if __name__ == "__main__":
    main()
