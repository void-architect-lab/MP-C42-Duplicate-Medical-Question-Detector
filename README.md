# MP-C42 — Duplicate Medical Question Detector

A small, local classification project for the Large Language Models course. It compares two medical questions and predicts whether they are **SIMILAR** (label 1) or **DIFFERENT** (label 0).

## What is implemented

- A course-style GPT-2 Small (124M) transformer implementation in `src/gpt_model.py`; no Hugging Face `pipeline`, `Trainer`, or `AutoModel` is used for the core model.
- Pretrained GPT-2 weights loaded into the custom PyTorch implementation.
- A new two-class classification head; the last transformer block, final layer norm, and classification head are fine-tuned.
- Curai Medical Question Pairs dataset, a reproducible stratified sample, train/validation/test split, seed 42, padding token 50256 and truncation.
- Before/after accuracy and F1, loss curve, confusion matrix, error examples, saved checkpoint, reload check, and a Gradio interface.
- `build_report.py` generates a Word report using the *measured results from your own run*.

**Medical safety:** this is an educational duplicate-question flagger, not a diagnosis tool and not suitable for patient-care decisions.

## Fast Windows run

Prerequisites: Windows 10/11, Python 3.11 recommended, internet access for first-time downloads, and about 3 GB of free space. An NVIDIA GPU is helpful but not mandatory (CPU training will be slower).

1. Open this folder in File Explorer.
2. Double-click `setup.bat` and let installation complete.
3. Double-click `run_train.bat`. First run downloads the 3048-row dataset and pretrained GPT-2 Small weights (~548 MB). The recommended training uses all 3,048 cleaned pairs, 4 epochs, sequence length 96, learning rate 2e-5 and random seed 42. It selects the checkpoint by validation loss and the decision threshold on validation data only.
4. When training completes, double-click `run_demo.bat`. Open the local address shown in the terminal (normally `http://127.0.0.1:7860`).
5. Take a screenshot of the working interface. Save it as `outputs/application_screenshot.png`, then regenerate the report by running:
   ```bat
   .venv\Scripts\python.exe build_report.py
   ```

Or from Command Prompt inside this folder:

```bat
setup.bat
run_train.bat
run_demo.bat
```

Each step is separate so a failed install can be diagnosed without restarting the training. The first model download needs an internet connection. If installation takes a long time, do not run multiple copies of the training process.

## Output files

After training, inspect:

- `outputs/results.json` — measured results and run configuration.
- `outputs/loss_curve.png` — training/validation loss.
- `outputs/confusion_matrix.png` — held-out test confusion matrix.
- `outputs/data_distribution.png` — sampled label counts.
- `outputs/test_predictions.csv` — every held-out prediction and correctness flag.
- `outputs/MP-C42_Report.docx` — generated Word report. It leaves student details and the final public URLs editable.
- `outputs/duplicate_detector.pt` — local model checkpoint. **Do not commit this file**; the assignment says not to upload model weights.

## Dataset

Curai Medical Question Pairs (MQP), 3,048 doctor-labelled question pairs. Source: https://github.com/curai/medical-question-pair-dataset . The project reads the headerless format `dr_id, question_1, question_2, label`; 1 means similar, 0 means different. The recommended run uses the full cleaned dataset with a reproducible stratified 70/15/15 split. The code prints and records the actual sample and split sizes. The final decision threshold is selected only on validation predictions; the test split is reserved for evaluation.

## Why LM Studio is not needed

LM Studio is designed mainly for running chat/completion models. This assessment asks for a custom GPT-2 classification head and measured before/after fine-tuning. The project therefore runs its own PyTorch model; it does not send medical questions to a third-party API or require LM Studio. The offline heuristic in `offline_demo.py` is only an emergency interface smoke test and is **not** the assessed model.

## Viva explanation (learn these)

- **Problem:** duplicate question detection; classify a pair as similar or different.
- **Tokenization:** GPT-2 byte-pair encoding through `tiktoken`; the two questions are formatted into one token sequence and truncated to 96 tokens.
- **Padding:** batch sequences are padded with token ID 50256. The classifier reads the logits at the final non-padding token.
- **Model:** GPT-2 Small pretrained backbone (12 transformer blocks, 768 embedding dimension, 12 attention heads) with a fresh 2-class linear head.
- **Training:** only the final transformer block, final layer norm and classification head are trainable, making the fine-tune smaller and faster.
- **Loss/optimizer:** cross-entropy and AdamW; the data split, seed, epochs and hyperparameters are recorded in `results.json`.
- **Metrics:** accuracy and F1 are measured on a held-out test split before and after fine-tuning. The best checkpoint is selected by validation loss, and the decision threshold is tuned on validation data only. The confusion matrix shows true vs predicted classes. No accuracy level is guaranteed; use the newly measured results.
- **Limitation:** similar wording is not always the same intent, and the dataset is small and English-focused. Outputs are not medical advice.

## AI-tools declaration

The author should disclose the use of AI coding assistance in the report as required by the course instructions, and should review and be able to explain every submitted line.

## GitHub: share the source safely

From Command Prompt in this folder, after confirming `.gitignore` excludes the model checkpoint:

```bat
git init
git add .
git status
git commit -m "Add MP-C42 duplicate medical question detector"
```

Create an **empty public repository** on GitHub in your browser (do not add a README, since one already exists locally), then replace `YOUR_GITHUB_USERNAME` below:

```bat
git branch -M main
git remote add origin https://github.com/YOUR_GITHUB_USERNAME/MP-C42-Duplicate-Medical-Question-Detector.git
git push -u origin main
```

If Git says your identity is unknown, set it once:

```bat
git config --global user.name "Your Name"
git config --global user.email "YOUR_GITHUB_EMAIL"
```

Before pushing, `git status --short` should **not** list `.venv`, `data/mqp.csv`, or `outputs/duplicate_detector.pt`. Do not force-add them. The repository shares source and real evaluation outputs; a friend who wants to reproduce training runs `setup.bat` and `run_train.bat` on their own computer. The model weights are downloaded to the local Hugging Face cache and the fine-tuned checkpoint stays local.

## Notebook link

The repository includes `MP-C42_Duplicate_Medical_Question_Detector.ipynb`. Once GitHub is public, open it in Colab using:
`https://colab.research.google.com/github/YOUR_GITHUB_USERNAME/MP-C42-Duplicate-Medical-Question-Detector/blob/main/MP-C42_Duplicate_Medical_Question_Detector.ipynb`

Replace the placeholder username/repository in the notebook/report and test both links in a private browser window before submitting.

## References

1. Curai, “Medical Question Pairs (MQP) Dataset,” GitHub repository, https://github.com/curai/medical-question-pair-dataset.
2. A. Radford et al., “Language Models are Unsupervised Multitask Learners,” OpenAI Technical Report, 2019.
3. S. Raschka, *Build a Large Language Model (From Scratch)*, Manning Publications, 2024; accompanying educational source code: https://github.com/rasbt/LLMs-from-scratch.
4. OpenAI Community, “GPT-2 model weights,” Hugging Face, https://huggingface.co/openai-community/gpt2.

## Hybrid demo post-processing

The interactive demo combines the fine-tuned GPT-2 classifier with a
transparent rule-based word/synonym consistency check. The heuristic can
override the GPT-2 label; therefore, the hybrid output is not a GPT-2-only
prediction. The reported 69.00% test accuracy and 0.6698 F1 refer to the
GPT-2 evaluation recorded by the training run, not to a separate evaluation
of the hybrid application. Both approaches can make mistakes.
