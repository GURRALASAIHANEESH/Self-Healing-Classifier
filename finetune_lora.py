from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, TaskType 
import torch
import os

# Load SST2 dataset (binary sentiment)
dataset = load_dataset("glue", "sst2")
checkpoint = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

# Tokenization
def preprocess(example):
    return tokenizer(example['sentence'], truncation=True, padding="max_length", max_length=128)

encoded_dataset = dataset.map(preprocess, batched=True)
encoded_dataset = encoded_dataset.rename_column("label", "labels")
encoded_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"]) 

# Load base model
base_model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)

# Apply LoRA
lora_config = LoraConfig(
    task_type=TaskType.SEQ_CLS,
    r=8,
    lora_alpha=16,
    lora_dropout=0.1,
    bias="none",
    target_modules=["q_lin", "v_lin"]  # key fix!
)

model = get_peft_model(base_model, lora_config)

# Training arguments
training_args = TrainingArguments(
    output_dir="./model",
    evaluation_strategy="epoch",
    logging_strategy="epoch",
    save_strategy="epoch",
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=4,
    learning_rate=2e-5,
    logging_dir="./logs",
    save_total_limit=1,
    load_best_model_at_end=True,
    report_to="none" 
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=encoded_dataset["train"].select(range(10000)),
    eval_dataset=encoded_dataset["validation"],
)

# Train
trainer.train()

# ✅ Add label mapping before saving
model.config.id2label = {0: "Negative", 1: "Positive"}
model.config.label2id = {"Negative": 0, "Positive": 1}

# Save final model
os.makedirs("model", exist_ok=True)
model.save_pretrained("model")
tokenizer.save_pretrained("model")

print("✅ Model fine-tuned and saved to ./model")
