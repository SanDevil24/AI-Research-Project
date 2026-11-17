import pandas as pd
import numpy as np
import evaluate
from datasets import Dataset
from transformers import (AutoTokenizer, AutoModelForSequenceClassification, 
                        TrainingArguments, Trainer)

train = pd.read_csv('../dataset/raw/train_multisocial_english_balanced_8000.csv')
train_dataset = train[['text', 'label']]

dataset = Dataset.from_dict(train_dataset).train_test_split(test_size=0.25)

model_checkpoint = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint, num_labels=2)

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

tokenized_datasets = dataset.map(tokenize_function, batched=True)

accuracy = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return accuracy.compute(predictions=predictions, references=labels)

training_args = TrainingArguments(
    output_dir = "./results",
    eval_strategy = "epoch",
    save_strategy = "epoch",
    learning_rate = 2e-5,
    per_device_train_batch_size = 8,
    per_device_eval_batch_size = 8,
    num_train_epochs = 3,
    weight_decay = 0.01,
    load_best_model_at_end = True
)

trainer = Trainer(
    model = model,
    args = training_args,
    train_dataset = tokenized_datasets["train"],
    eval_dataset = tokenized_datasets["test"],
    compute_metrics = compute_metrics
)

trainer.train()

model.save_pretrained("model")
tokenizer.save_pretrained("model")
