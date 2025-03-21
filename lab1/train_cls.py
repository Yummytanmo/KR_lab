from BERT import DistilBertForSequenceClassification
from transformers import AutoTokenizer, TrainingArguments, Trainer
from datasets import load_dataset
import numpy as np
import evaluate
from datasets import DownloadMode
import os

# Set environment variable for cache directory
os.environ["HF_HOME"] = "./cache"

id2label = {0: "NEGATIVE", 1: "POSITIVE"}
label2id = {"NEGATIVE": 0, "POSITIVE": 1}

tokenizer = AutoTokenizer.from_pretrained("./cache/distilbert")
model = DistilBertForSequenceClassification.from_pretrained("./cache/distilbert", num_labels=2, id2label=id2label, label2id=label2id)

dataset = load_dataset("stanfordnlp/sst2", cache_dir="./cache", download_mode=DownloadMode.REUSE_DATASET_IF_EXISTS)
accuracy = evaluate.load("accuracy", cache_dir="./cache")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    res = accuracy.compute(predictions=predictions, references=labels)
    print(res)
    return res


def tokenize_function(examples):
    return tokenizer(examples["sentence"], truncation=True)

tokenized_datasets = dataset.map(tokenize_function, batched=True)

training_args = TrainingArguments(
    output_dir="./ckpt/CLS_ckpt",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    evaluation_strategy="epoch",
    weight_decay=0.01,
    save_strategy="epoch",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

trainer.train()
eval_results = trainer.evaluate()
print(f"Validation Accuracy: {eval_results['eval_accuracy']}")