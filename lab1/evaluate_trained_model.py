from transformers import AutoTokenizer, AutoModelForTokenClassification, Trainer
import numpy as np
import evaluate
from datasets import load_dataset

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("./cache/distilbert")
model = AutoModelForTokenClassification.from_pretrained("./ckpt/NER_ckpt")

# Load the dataset
dataset = load_dataset("conll2003", cache_dir="./cache", split="test")

# Define the label list and seqeval
label_list = ['O', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-LOC', 'I-LOC', 'B-MISC', 'I-MISC']
seqeval = evaluate.load("seqeval")

# Define the compute_metrics function
def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    results = seqeval.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }

# Tokenize the dataset
def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(examples["tokens"], truncation=True, is_split_into_words=True)

    labels = []
    for i, label in enumerate(examples["ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx])
            else:
                label_ids.append(-100)
            previous_word_idx = word_idx
        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs

tokenized_dataset = dataset.map(tokenize_and_align_labels, batched=True)

# Initialize the Trainer
trainer = Trainer(
    model=model,
    tokenizer=tokenizer,
)

# Evaluate the model
metrics = trainer.evaluate(eval_dataset=tokenized_dataset, compute_metrics=compute_metrics)
print(metrics)
