import pandas as pd
from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer,
)
import numpy as np
import evaluate
import torch

CSV_PATH = "WELFake_Dataset.csv"
TEXT_COL = "text"     
LABEL_COL = "label" 
df = pd.read_csv(CSV_PATH)
df = df.dropna(subset=["text"])
df["text"] = df["text"].astype(str)

if df["label"].dtype == "object":
    all_labels = sorted(df["label"].unique())
    label2id = {lab: i for i, lab in enumerate(all_labels)}
    id2label = {i: lab for lab, i in label2id.items()}
    df["label"] = df["label"].map(label2id)
else:
    unique_labels = sorted(df["label"].unique())
    remap = {old: new for new, old in enumerate(unique_labels)}
    df["label"] = df["label"].map(remap)
    unique_labels = sorted(df["label"].unique())
    id2label = {int(i): str(i) for i in unique_labels}
    label2id = {str(i): int(i) for i in unique_labels}

id2label = {int(k): str(v) for k, v in id2label.items()}
label2id = {str(k): int(v) for k, v in label2id.items()}
num_labels = len(id2label)
train_df, test_df = train_test_split(
    df,
    test_size=0.2,
    random_state=42,
    stratify=df["label"],
)
train_dataset = Dataset.from_pandas(train_df.reset_index(drop=True))
test_dataset = Dataset.from_pandas(test_df.reset_index(drop=True))
datasets = DatasetDict({
    "train": train_dataset,
    "test": test_dataset,
})

model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)

max_length = 256

def preprocess_function(examples):
    texts = [str(t) for t in examples["text"]]
    return tokenizer(
        texts,
        padding=False, 
        truncation=True,
        max_length=max_length,
    )

cols_to_keep = ["text", "label"]
remove_cols_train = [c for c in datasets["train"].column_names if c not in cols_to_keep]
remove_cols_test = [c for c in datasets["test"].column_names if c not in cols_to_keep]

tokenized_train = datasets["train"].map(
    preprocess_function,
    batched=True,
    remove_columns=remove_cols_train,
)
tokenized_test = datasets["test"].map(
    preprocess_function,
    batched=True,
    remove_columns=remove_cols_test,
)

tokenized_datasets = DatasetDict({
    "train": tokenized_train,
    "test": tokenized_test,
})
tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=num_labels,
    id2label=id2label,
    label2id=label2id,
)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
accuracy = evaluate.load("accuracy")
f1 = evaluate.load("f1")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy.compute(predictions=preds, references=labels)["accuracy"],
        "f1": f1.compute(predictions=preds, references=labels, average="weighted")["f1"],
    }

batch_size = 16
training_args = TrainingArguments(
    output_dir="./bert-fake-news",
    num_train_epochs=3,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    learning_rate=2e-5,
    weight_decay=0.01,
    logging_steps=50,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

def predict_label(text: str):
    model.eval()
    enc = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
    )
    with torch.no_grad():
        outputs = model(**enc)
        logits = outputs.logits
        pred_id = int(torch.argmax(logits, dim=-1))
    return {
        "label_id": pred_id,
        "label": id2label[int(pred_id)],
        "logits": logits.tolist(),
    }

def main():
    trainer.train()
    metrics = trainer.evaluate()
    print("Test metrics:", metrics)
    save_dir = "./bert-fake-news-checkpoint"
    trainer.save_model(save_dir)
    tokenizer.save_pretrained(save_dir)

if __name__ == "__main__":
    main()

#Test metrics: {'eval_loss': 0.0328199602663517, 'eval_accuracy': 0.9947291767806367, 'eval_f1': 0.9947293544229745, 'eval_runtime': 148.4915, 'eval_samples_per_second': 97.103, 'eval_steps_per_second': 6.074, 'epoch': 3.0}