<div align="center">

# 🎙️ Uyghur Automatic Speech Recognition (ASR) 🗣️
### 🌍 Low-Resource Speech-to-Text with `facebook/mms-300m` + CTC

**Deep Learning Practice (DLP) · IIT Madras BS Degree Program**

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![HuggingFace](https://img.shields.io/badge/🤗_Transformers-FFD21E?style=for-the-badge&logoColor=black)
![Kaggle](https://img.shields.io/badge/Kaggle-20BEFF?style=for-the-badge&logo=kaggle&logoColor=white)
![Task](https://img.shields.io/badge/Task-Speech_Recognition-success?style=for-the-badge)
![Model](https://img.shields.io/badge/Model-MMS--300M-blueviolet?style=for-the-badge)
![Metric](https://img.shields.io/badge/Metric-CER-orange?style=for-the-badge)
![Val CER](https://img.shields.io/badge/Val_CER-0.0536-brightgreen?style=for-the-badge)

</div>

---

## 🎯 Problem Statement

Build a **high-performing ASR system** that listens to an audio clip in **Uyghur** 🗣️ and outputs the correct **text transcription** 📝.

- 🎧 **23.95 hours** of Uyghur speech (9,468 mono `.wav` clips @ 16 kHz)
- 🆓 **No restrictions** on models, compute or external data
- 🌱 A genuinely **low-resource language** setting — perfect for multilingual pre-trained speech models

---

## 📦 Dataset

| 📁 File | 🧾 Columns | 🔢 Size |
|---|---|---|
| `train.csv` | `ID`, `filepath`, `transcription` | **7,574** clips |
| `test.csv` | `ID`, `filepath` | **1,894** clips |
| `sample.csv` | `ID`, `transcription` | Submission template |

✂️ Internal split used: **6,816 train** / **758 validation** (90/10, seed 42)

---

## 📏 Evaluation Metric — Character Error Rate (CER)

$$CER = \frac{S + D + I}{N}$$

| Symbol | Meaning |
|---|---|
| 🔁 `S` | Substitutions |
| ➖ `D` | Deletions |
| ➕ `I` | Insertions |
| 🔤 `N` | Characters in the reference transcription |

📉 **Lower is better** — `0` means a perfect transcription.

---

## 🏆 Results

| 🧪 Model | 🔧 Head | 📉 Best Validation CER |
|---|---|---|
| `facebook/mms-300m` | CTC | **0.0536** 🎉 |

📈 **Validation CER progress (every 2 epochs)**

| Epoch | 2 | 4 | 6 | 8 | 10 | 12 | 14 | 16 | 18 |
|---|---|---|---|---|---|---|---|---|---|
| CER | 0.1849 | 0.0870 | 0.0731 | 0.0670 | 0.0615 | 0.0590 | 0.0571 | 0.0554 | **0.0536** |

> 🥇 Leaderboard CER: _add your final Kaggle score here_

---

## 🧠 Approach

| 🎛️ Choice | ✅ What I used | 💭 Why |
|---|---|---|
| 🤖 **Model** | `facebook/mms-300m` | Multilingual pre-training with native Uyghur (`uig`) coverage — a strong fit for a low-resource language |
| 🎯 **Head** | CTC | Lighter & faster than seq2seq, natural for character-level targets |
| 🔤 **Tokenizer** | Custom character-level vocab (37 tokens) | Built from **training transcriptions only** — small Latin character set, no subword tokenizer needed |
| ⚡ **Precision** | Mixed precision (AMP) + gradient checkpointing | Fits comfortably on a Kaggle GPU |
| 🧊 **Frozen part** | CNN feature encoder | Standard wav2vec2 / MMS fine-tuning recipe |
| ⏱️ **Time budget** | 7.8h training cap, 8.5h hard stop | Guarantees a valid `submission.csv` is always produced |

---

## 🛠️ Pipeline

```text
🎧 Audio (.wav)
   │
   ▼
🔊 Load → mono → resample to 16 kHz → crop to 30s max
   │
   ▼
📐 Wav2Vec2FeatureExtractor (normalised waveform)
   │
   ▼
🧠 MMS-300M encoder  (feature encoder frozen)
   │
   ▼
🎯 CTC head over 37-char vocab
   │
   ▼
🔍 Greedy CTC decoding  ( | → space )
   │
   ▼
📝 Transcription  →  📤 submission.csv  (ID, transcription)
```

### ⚙️ Hyperparameters

| Param | Value |
|---|---|
| 🎓 Optimizer | AdamW (weight decay 0.01) |
| 📈 Learning rate | `1e-4` with cosine schedule |
| 🔥 Warmup steps | 150 |
| 📦 Batch | 4 × 8 grad-accumulation = **32** effective |
| 🔁 Max epochs | 30 (time-budget bounded) |
| 🛑 Early stopping | Patience 3 (validated every 2 epochs) |
| ✂️ Gradient clipping | 1.0 |
| 🔢 Parameters | 315.5M total · 311.3M trainable |
| 🎲 Seed | 42 |

---

## 🌟 Key Features

- 🔤 **Vocab from train only** — no test leakage
- ⏱️ **Time-budget-aware training loop** — checked before each epoch and every 100 steps
- 💾 **Best-checkpoint saving** based on validation CER, not the last epoch
- 🛡️ **Fallback checkpoint** so inference always runs
- ✅ **Built-in submission verification** — column names, row count, ID order, no NaNs
- 🧪 **Dry-run sanity check** before committing hours of training

---

## 💡 Key Learnings

- 🌐 **Multilingual pre-training matters** — MMS covers Uyghur natively, which is a big edge over general models for this language.
- 🎯 **CTC is a great fit** for character-level, low-resource ASR.
- 📉 **CER kept improving steadily** through epoch 18 — more training time was still paying off.
- ⏳ **Plan for the compute budget** — a time-aware loop avoids losing a run to a hard session limit.

---

## 🔮 Future Work

- 📖 Add an **n-gram / KenLM language model** to CTC decoding
- 🎚️ **Data augmentation** (speed perturbation, SpecAugment, noise)
- 🤝 **Ensembling** multiple checkpoints or models
- 🔍 Compare with **Whisper** and larger MMS variants (`mms-1b`)
- 🧮 **Beam search** decoding instead of greedy

---

## 🚀 How to Run

1. 📥 Open the notebook on **Kaggle**
2. ➕ Add the competition dataset
3. ⚙️ Settings → Accelerator → **GPU** 🎮
4. ▶️ **Run All** — best model is saved to `/kaggle/working/best_mms` and predictions to `submission.csv`

```bash
pip install transformers jiwer soundfile torchaudio
```

⏱️ Expect roughly **~23–25 min/epoch** on a Kaggle GPU.

---

## 🗂️ Repository Structure

```text
📦 DLP-Uyghur-ASR-MMS-CTC
 ┣ 📔 dlp-uyghur-asr-mms300m-ctc.ipynb
 ┗ 📄 README.md
```

---

<div align="center">

### 👨‍💻 Author
**Saini** · (CyberSoul) 🎓

⭐ If this helped you, drop a star on the repo! ⭐

*Made with ❤️, ☕ and a lot of GPU hours* 🔥

</div>
