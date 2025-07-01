# NLP-Based Multitask Analysis and TTS-RVC Integration

This project implements a multitask NLP pipeline for **topic classification**, **sentiment analysis**, and **summarization** of Turkish conversational data. It also integrates a **TTS (Text-to-Speech)** and **RVC (Retrieval-Based Voice Conversion)** pipeline to convert summaries into speech with voice modulation.

---

## 📌 Table of Contents
- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Usage](#usage)
  - [Multitask NLP Pipeline](#multitask-nlp-pipeline)
  - [TTS and RVC Voice Conversion](#tts-and-rvc-voice-conversion)
- [Models](#models)
- [Dataset](#dataset)
- [Notes](#notes)
- [License](#license)

---

## 🚀 Features

- **Multitask NLP Pipeline**
  - Topic Classification
  - Sentiment Analysis (Zero-Shot + Word-Based)
  - Summarization of Conversations
- **TTS and RVC Integration**
  - Converts summarized text into speech using TTS
  - Modulates voice using RVC
- **Gradio UI**
  - Interactive tabs for analysis and audio generation

---

## 🛠 Requirements

- Python 3.10
- Compatible CUDA version (optional)
- Visual Studio Build Tools:
  - MSVC v142 - VS 2019 C++ x64/x86 build tools
  - Windows 10 SDK
  - C++ CMake tools
- NVIDIA GPU with CUDA (recommended)

---

## ⚙️ Installation

```bash
# Step 1: Create virtual environment
python -m venv .venv

# Step 2: Activate
# Linux/Mac:
source .venv/bin/activate
# Windows:
.venv\Scripts\activate

# Step 3: Install dependencies
pip install -r Requirements.txt
```

---

## 🗂 Project Structure

```plaintext
project-root/
│
├── methods/
│   └── topic_sentiment_summarization.py
│
├── tts_rvc_logic.py
├── tts_rvc_ui.py
├── Requirements.txt
├── .gitignore
└── README.md
```

> ⚠️ **Excluded from repository:**
> - `datasets/` → Please check the 📁 Dataset section
> - `models/`, `local_models/` → Downloaded from Hugging Face
> - `rvc_models/` → `.pth` files (manual)
> - `outputs/` → Generated results

---

## 💻 Usage

### 🧠 Multitask NLP Pipeline

```bash
python methods/topic_sentiment_summarization.py
```

- Input: Custom JSON dataset
- Output: `multitask_output.json`, `summarized_output.json` in `outputs/`

### 🔊 TTS and RVC Voice Conversion

```bash
python tts_rvc_ui.py
```

- **Dataset Analysis Tab:** Upload conversation JSON
- **Voice Conversion Tab:** Choose an RVC model, adjust pitch, generate audio

---

## 🧠 Models

| Task                | Model                               | Link                                                                 |
|---------------------|--------------------------------------|----------------------------------------------------------------------|
| Topic Classification | `bert-topic-classification-turkish` | [Hugging Face](https://huggingface.co/GosamaIKU/bert-topic-classification-turkish) |
| Sentiment Analysis   | `xlm-roberta-large-xnli`            | [Hugging Face](https://huggingface.co/joeddav/xlm-roberta-large-xnli) |
| Text-to-Speech       | `facebook/mms-tts-tur`              | [Hugging Face](https://huggingface.co/facebook/mms-tts-tur) |
| RVC Models           | `.pth` files                        | Download manually, place in `rvc_models/` folder |

---

- **RVC Sample Model**:
  - A sample `.pth` voice conversion model is included in `rvc_models/`.
  - You can replace it with your own model.



## 📁 Dataset


The dataset is hosted on Kaggle:

🔗 **[Download from Kaggle](https://www.kaggle.com/datasets/anills/turkish-call-center-conversations/data)** 

Expected format:

```json
{
  "conversations": [
    {
      "speaker": "customer",
      "text": "Merhaba, kargom hâlâ gelmedi."
    },
    {
      "speaker": "agent",
      "text": "Hemen kontrol ediyorum."
    }
  ]
}
```

---

## 📝 Notes

- All necessary models must be downloaded manually from Hugging Face and placed in the correct folders (`models/`, `local_models/`, `rvc_models/`).
- Output files (such as `.json` and audio `.wav` or `.mp3`) will be automatically generated when running the pipeline.
- For practical and size-related reasons, **datasets**, **models**, and **outputs** are excluded from this repository. You can access the dataset from the [Kaggle link](#dataset), and download models from the [Models](#models) section.

---

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
