# 🤖 Self-Healing Classifier (LangGraph + Fine-Tuned DistilBERT)

## 📌 Overview
This project builds a **self-healing sentiment classifier** using a **fine-tuned DistilBERT model** integrated into a **LangGraph DAG**. The system detects low-confidence predictions and invokes a fallback mechanism (clarification or backup model) to improve robustness and accuracy.

---

## 🚀 Project Build Steps

### Step 1: Clone Repository and Prepare Files
```bash
git clone https://github.com/GURRALASAIHANEESH/Self-Healing-Classifier.git
cd Self-Healing-Classifier
```

### Step 2: Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate      # On Linux/Mac
venv\Scripts\activate         # On Windows
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```
Packages include:
```txt
transformers
datasets
peft
torch
matplotlib
langgraph
huggingface_hub
```

---

### Step 4: Fine-Tune the Model
Run the following to fine-tune DistilBERT on the SST-2 dataset:
```bash
python fine_tune.py
```
This script:
- Loads the SST-2 dataset
- Tokenizes using `AutoTokenizer`
- Applies LoRA for parameter-efficient fine-tuning
- Trains the model for 4 epochs
- Saves the model and tokenizer in `./model`

After training, assign readable labels:
```python
model.config.id2label = {0: "Negative", 1: "Positive"}
model.config.label2id = {"Negative": 0, "Positive": 1}
model.save_pretrained("./model")
tokenizer.save_pretrained("./model")
```

---

### Step 5: Set Up LangGraph Workflow
Each node performs a distinct role:
- `InferenceNode`: Runs predictions with the fine-tuned model
- `ConfidenceCheckNode`: Evaluates confidence threshold
- `FallbackNode`: Handles user clarification or backup model fallback
- `FinalOutputNode`: Logs and returns final decision

All node implementations are placed in the `nodes/` folder.

---

### Step 6: Run the CLI Classifier
Start the DAG with:
```bash
python dag_main.py
```
This CLI:
- Accepts text input
- Classifies it using the DAG workflow
- Triggers fallback if confidence is low
- Logs all decisions
- Displays final label and plots fallback/accuracy trends

---

## 📆 Why the Model Isn’t on GitHub
The fine-tuned model contains large weight files (e.g., `.pt`, `.safetensors`, optimizers) that exceed GitHub’s 100MB file limit. Instead, the full model is hosted on Hugging Face.

### ✅ Access the Full Model Here:
**[Hugging Face Model](https://huggingface.co/sunnypatel782/self-healing-sentiment-model/tree/main)**

Includes:
- Fine-tuned weights
- Tokenizer configs
- Adapter model

---

## 📁 Folder Structure
```
├── dag_main.py             # LangGraph DAG CLI
├── fine_tune.py            # Fine-tuning script
├── model/                  # Model output folder (not pushed to GitHub)
├── logfile.log             # Prediction & fallback logs
├── confidence_trend.png    # Trend plot
├── logging_utils.py        # Event logging
├── nodes/
│   ├── confidence_check.py
│   ├── inference_node.py
│   ├── fallback_node.py
│   └── final_output_node.py
├── requirements.txt
├── README.md
```

---

## 📨 Submission Instructions
The Google Form asks for:
- GitHub repo
- Public drive link for ZIP file
- Demo video

### ✅ GitHub Repo (Code Only)
> https://github.com/GURRALASAIHANEESH/Self-Healing-Classifier

### ✅ Hugging Face Model
> https://huggingface.co/sunnypatel782/self-healing-sentiment-model

### ✅ Google Drive (ZIP with model, code, logs, video)
> Replace with your actual Google Drive public link before submission

---

## 🔄 Sample CLI Interaction
```
Enter text to classify (or 'exit'): the hero acting was a disaster
[InferenceNode] Predicted label: Negative | Confidence: 83.48%
[ConfidenceCheckNode] Confidence too low. Triggering fallback...
[FallbackNode] Could you clarify your intent? Was this a Positive review?
User: No
✅ Final Label: Negative (Confirmed by user)
```

---

## ✅ Deliverables Checklist
- [x] Fine-tuned model hosted on Hugging Face
- [x] LangGraph DAG with fallback + backup model
- [x] CLI with logging and confidence chart
- [x] GitHub repo for code
- [x] Google Drive ZIP with everything included
- [x] Demo video walkthrough

---

Ready to submit! 🚀
