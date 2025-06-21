# ✅ Complete and Robust dag_main.py (Final Version with Bonus Features)

from typing import TypedDict
from langgraph.graph import StateGraph
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import torch
from nodes.confidence_check import check_confidence
from logging_utils import log_event
import logging
import matplotlib.pyplot as plt

# Load model & tokenizer from local fine-tuned directory
model_path = "./model"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)
model.config.id2label = {0: "Negative", 1: "Positive"}
model.config.label2id = {"Negative": 0, "Positive": 1}
model.eval()

# Zero-shot backup classifier
zero_shot = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

def backup_classify(text):
    result = zero_shot(text, candidate_labels=["Positive", "Negative"])
    return result['labels'][0], result['scores'][0]

# Track confidence and fallback stats
confidence_log = []
stats = {
    "total_inputs": 0,
    "fallback_triggered": 0
}

class ClassifierState(TypedDict):
    input: str
    label: str
    confidence: float
    model_output: dict
    decision: str
    final_label: str

def inference_node(state: ClassifierState) -> ClassifierState:
    text = state["input"]
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=1)
        confidence, predicted_class = torch.max(probs, dim=1)

    label = model.config.id2label.get(predicted_class.item(), f"LABEL_{predicted_class.item()}")
    confidence = confidence.item()

    confidence_log.append(confidence)
    log_event("InferenceNode", f"Predicted label: {label} | Confidence: {confidence*100:.2f}%")
    print(f"[InferenceNode] Predicted label: {label} | Confidence: {confidence*100:.2f}%")

    return {
        **state,
        "label": label,
        "confidence": confidence,
        "model_output": {"label": label, "confidence": confidence},
    }

def confidence_check_node(state: ClassifierState) -> ClassifierState:
    stats["total_inputs"] += 1
    result = check_confidence(state["model_output"], threshold=0.95)
    decision = result["decision"]

    if decision == "accept":
        print("[ConfidenceCheckNode] Confidence is sufficient. Accepting result.")
        final_label = state["label"]
        log_event("ConfidenceCheckNode", f"Accepted model prediction → Final Label: {final_label}")
        return {**state, "decision": decision, "final_label": final_label}
    else:
        print("[ConfidenceCheckNode] Confidence too low. Triggering fallback...")
        stats["fallback_triggered"] += 1
        return {**state, "decision": decision}

def fallback_node(state: ClassifierState) -> ClassifierState:
    predicted_label = state["label"]
    opposite_label = "Positive" if predicted_label == "Negative" else "Negative"

    print(f"[FallbackNode] Could you clarify your intent? Was this a {opposite_label} review?")
    user_input = input("User: ").lower().strip()

    yes_variants = ["yes", "yeah", "yep", "definitely", "of course", "sure"]
    no_variants = ["no", "nope", "nah", "not really", "no way"]

    if any(word in user_input for word in yes_variants):
        final_label = opposite_label
        print(f"✅ Final Label: {final_label} (Corrected via user clarification)")
        log_event("FallbackNode", f"Corrected via user input → Final Label: {final_label}")
    elif any(word in user_input for word in no_variants):
        final_label = predicted_label
        print(f"✅ Final Label: {final_label} (Confirmed by user)")
        log_event("FallbackNode", f"User confirmed model prediction → Final Label: {final_label}")
    else:
        backup_label, backup_conf = backup_classify(state["input"])
        final_label = backup_label
        print(f"🤖 Backup Model Suggests: {final_label} ({backup_conf*100:.2f}%)")
        print(f"✅ Final Label: {final_label} (Used backup model)")
        log_event("FallbackNode", f"Used backup model → Final Label: {final_label}")

    return {**state, "final_label": final_label}

def final_output_node(state: ClassifierState) -> ClassifierState:
    print(f"\n✅ Final Label: {state['final_label']}")
    log_event("FinalOutputNode", f"Final Label: {state['final_label']}")
    return state

def route_decision(state: ClassifierState) -> str:
    return state["decision"]

def display_confidence_curve():
    if not confidence_log:
        return
    plt.plot(confidence_log, marker='o')
    plt.title("Confidence Over Inputs")
    plt.ylabel("Confidence")
    plt.xlabel("Input Number")
    plt.grid(True)
    plt.savefig("confidence_trend.png")
    print("📊 Saved confidence trend to confidence_trend.png")

def print_stats():
    if stats["total_inputs"] == 0:
        return
    print("\n📊 Summary Stats:")
    print(f"Total Inputs: {stats['total_inputs']}")
    print(f"Fallback Triggered: {stats['fallback_triggered']}")
    print(f"Fallback Rate: {(stats['fallback_triggered'] / stats['total_inputs']) * 100:.2f}%")

def main():
    print("🤖 Self-Healing Classifier (LangGraph + Fine-Tuned DistilBERT)")
    print("-" * 61)

    builder = StateGraph(state_schema=ClassifierState)
    builder.add_node("InferenceNode", inference_node)
    builder.add_node("ConfidenceCheckNode", confidence_check_node)
    builder.add_node("FallbackNode", fallback_node)
    builder.add_node("FinalOutputNode", final_output_node)

    builder.set_entry_point("InferenceNode")
    builder.add_edge("InferenceNode", "ConfidenceCheckNode")
    builder.add_conditional_edges(
        "ConfidenceCheckNode",
        route_decision,
        {
            "accept": "FinalOutputNode",
            "trigger_fallback": "FallbackNode"
        }
    )
    builder.add_edge("FallbackNode", "FinalOutputNode")

    graph = builder.compile()

    while True:
        user_input = input("\nEnter text to classify (or 'exit'): ").strip()
        if user_input.lower() == "exit":
            display_confidence_curve()
            print_stats()
            print("👋 Goodbye!")
            break

        initial_state: ClassifierState = {
            "input": user_input,
            "label": "",
            "confidence": 0.0,
            "model_output": {},
            "decision": "",
            "final_label": ""
        }

        graph.invoke(initial_state)

if __name__ == "__main__":
    main()
