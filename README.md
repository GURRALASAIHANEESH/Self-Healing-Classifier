# 🤖 Self-Healing Classifier (LangGraph + Fine-Tuned DistilBERT)

## 📌 Overview
This project builds a **self-healing sentiment classifier** using a **fine-tuned DistilBERT model** integrated into a **LangGraph DAG**. The system detects low-confidence predictions and invokes a fallback mechanism (clarification or backup model) to improve robustness and accuracy.

---

## 🚀 Project Build & Run Instructions

### Step 1: Clone the Repository
```bash
git clone https://github.com/GURRALASAIHANEESH/Self-Healing-Classifier.git
cd Self-Healing-Classifier
```

### Step 2: Set Up Environment
Create a virtual environment and install dependencies:
```bash
python -m venv venv
# On Linux/macOS:
source venv/bin/activate
# On Windows:
venv\Scripts\activate

pip install -r requirements.txt
```

### Step 3: Download Fine-Tuned Model
Due to GitHub's file size limits, the fine-tuned model (980MB+) is **not included in the GitHub repo**.
Instead, download the model from Hugging Face:
> [Hugging Face Model](https://huggingface.co/sunnypatel782/self-healing-sentiment-model)

**Place the downloaded model files into the `./model/` directory.**

### Step 4: Run the CLI
Execute the DAG-based CLI interface:
```bash
python dag_main.py
```

This launches the interactive classifier where you:
- Input a sentence
- Get prediction and confidence
- If confidence is low → fallback is triggered
- User clarification or backup model is invoked
- Final label is shown and logged

---

## 📆 Folder Structure
```
├── dag_main.py             # Main CLI & LangGraph DAG runner
├── fine_tune.py            # Script for fine-tuning DistilBERT
├── model/                  # Place Hugging Face model files here
├── logfile.log             # Logs of interactions and decisions
├── confidence_trend.png    # Confidence graph over inputs
├── logging_utils.py        # Timestamped logger
├── upload_model.py         # Script to push model to Hugging Face
├── nodes/
│   ├── confidence_check.py
│   ├── inference_node.py
│   ├── fallback_node.py
│   └── final_output_node.py
├── requirements.txt
├── README.md
```

---

## 🧠 Bonus Feature: Backup Model Trigger
This project implements the **optional bonus** requirement:
- If the base model is **not confident** _and_ user clarification is vague or absent,
- It **falls back to a backup zero-shot sentiment classifier**.

### 🔁 Example Triggers for Backup:
- Vague reviews: _"It was fine I guess..."
- Sarcastic or confusing inputs
- User replies "maybe", "kinda", etc.

---

## 🔮 Example Output with Bonus in Action
```
Enter text to classify (or 'exit'): the heroin was hot
[InferenceNode] Predicted label: Negative | Confidence: 84.55%
[ConfidenceCheckNode] Confidence too low. Triggering fallback...
[FallbackNode] Could you clarify your intent? Was this a Positive review?
User: yap
🤖 Backup Model Suggests: Positive (79.98%)
✅ Final Label: Positive (Used backup model)
```

---

## 📩 Submission Links
### GitHub Repository (Code only):
> https://github.com/GURRALASAIHANEESH/Self-Healing-Classifier

### Hugging Face Model (Due to GitHub size limits):
> https://huggingface.co/sunnypatel782/self-healing-sentiment-model

### Google Drive Folder (Full Project Zip + Video Demo):
> [G Sai Haneesh _ATG Technical Assignment](https://drive.google.com/drive/folders/1usHvAxPVfVC4kWBTg1Kdf03jmNv_C8pR?usp=drive_link)

---

## 🎥 Demo Video
> [Click to Watch the Demo](https://drive.google.com/file/d/1lsanQj8agSGo_NKgK4OjPyPCzcgoYlES/view?usp=drive_link)

---

## ✅ Final Deliverables Checklist
- [x] Fine-tuned transformer model (Hugging Face)
- [x] LangGraph DAG pipeline
- [x] Confidence-based fallback system
- [x] Bonus backup model integration
- [x] Structured CLI + logging + confidence graph
- [x] GitHub with clean modular code
- [x] Google Drive (zip + video demo)
