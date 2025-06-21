# 🤖 Self-Healing Classifier (LangGraph + Fine-Tuned DistilBERT)

## 📌 Overview
This project builds a **self-healing sentiment classifier** using a **fine-tuned DistilBERT model** integrated in a **LangGraph DAG**. The system detects low-confidence predictions and invokes a fallback mechanism (clarification or backup model) to improve reliability.

---

## 🚀 Project Build Steps

### Step 1: Environment Setup
Install required packages:
```bash
pip install -r requirements.txt
```
Make sure you have:
```txt
transformers
datasets
peft
torch
matplotlib
langgraph
```

---

### Step 2: Fine-Tune the Model
Run the following to fine-tune DistilBERT on SST-2 dataset:
```bash
python fine_tune.py
```
This script:
- Loads SST-2 dataset
- Tokenizes using `AutoTokenizer`
- Applies LoRA for efficient tuning
- Trains for 4 epochs
- Saves model and tokenizer to `./model`

Manually assign readable labels (optional but recommended):
```python
model.config.id2label = {0: "Negative", 1: "Positive"}
model.config.label2id = {"Negative": 0, "Positive": 1}
model.save_pretrained("./model")
tokenizer.save_pretrained("./model")
```

---

### Step 3: Define LangGraph Nodes
Each node in the DAG performs a distinct role:
- `InferenceNode`: Runs prediction with confidence score
- `ConfidenceCheckNode`: Triggers fallback if confidence is low
- `FallbackNode`: Asks user to confirm or correct label (or use backup model)
- `FinalOutputNode`: Logs and displays final classification

Code for each node resides in the `nodes/` folder.

---

### Step 4: Run the DAG Classifier
Run the LangGraph-based CLI system using:
```bash
python dag_main.py
```
This script handles:
- User input loop
- Calling the model for predictions
- Confidence checks and fallback logic
- Structured logging and statistics

---

## 🔖 Sample CLI Session (Full Flow with Fallback & Backup Model)
```
Enter text to classify (or 'exit'): The Acting of hero was disaster
[InferenceNode] Predicted label: Negative | Confidence: 83.48%
[ConfidenceCheckNode] Confidence too low. Triggering fallback...
[FallbackNode] Could you clarify your intent? Was this a Positive review?
User: No
🚀 Final Label: Negative (Confirmed by user)

---

Enter text to classify (or 'exit'): the heroin was hot
[InferenceNode] Predicted label: Negative | Confidence: 84.55%
[ConfidenceCheckNode] Confidence too low. Triggering fallback...
[FallbackNode] Could you clarify your intent? Was this a Positive review?
User: yap
🤖 Backup Model Suggests: Positive (79.98%)
🚀 Final Label: Positive (Used backup model)

---

Enter text to classify (or 'exit'): do you think movie was good
[InferenceNode] Predicted label: Positive | Confidence: 91.01%
[ConfidenceCheckNode] Confidence too low. Triggering fallback...
[FallbackNode] Could you clarify your intent? Was this a Negative review?
User: yes
🚀 Final Label: Negative (Corrected via user clarification)

📊 Summary Stats:
Total Inputs: 9
Fallback Triggered: 4
Fallback Rate: 44.44%
📈 Confidence trend saved to confidence_trend.png
```

---

## 📊 Logs & Visualization
- Logs stored in `logfile.log` with timestamps, confidence, and labels
- Confidence trend plot auto-generated in `confidence_trend.png`
- Fallback frequency stats shown at the end of session

---

## 📹 Demo Video Checklist
- Start CLI session with high and low confidence inputs
- Show fallback clarification example
- Display saved confidence curve
- Explain DAG node flow and how fallback logic works

---

## 📁 Project Structure
```
├── dag_main.py             # LangGraph DAG main script
├── fine_tune.py            # Fine-tune DistilBERT with LoRA
├── model/                  # Saved fine-tuned model
├── logfile.log             # Prediction + fallback logs
├── confidence_trend.png    # Confidence curve plot
├── logging_utils.py        # Custom logger for events
├── nodes/
│   ├── confidence_check.py
│   ├── inference_node.py
│   ├── fallback_node.py
│   └── final_output_node.py
├── requirements.txt
├── README.md
```

---

## ✅ Deliverables Checklist
- [x] Fine-tuned model with readable labels
- [x] Source code (training + DAG)
- [x] CLI with fallback logic
- [x] Log file & confidence visuals
- [x] Clear README with steps
- [x] Demo video ✅

---

## 📬 Submission Format
Submit as:
- GitHub repo **OR** zipped folder containing:
  - All source code
  - Fine-tuned model (`model/`)
  - `README.md`
  - `logfile.log` + plots
  - Demo video or its YouTube link

---

## 👨‍💻 Author
Submitted for **ATG Machine Learning Intern Assignment**.
Questions? Reach out at `sunny.patel@demoemail.com` (replace with your actual email).

---

All set. Happy submitting! 🚀
