import pandas as pd
from sklearn.model_selection import train_test_split
from datasets import Dataset, DatasetDict, load_from_disk

tokenized_datasets = load_from_disk("./tokenized_datasets-no_diy-400000")
train_dataset = tokenized_datasets["train"].shuffle(seed=42)
eval_dataset = tokenized_datasets["eval"].shuffle(seed=42)

from transformers import AutoModelForSequenceClassification
model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2, hidden_dropout_prob=0.2)

from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import numpy as np
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average="binary")
    return {
        "eval_loss": float(np.mean(logits)),  # 确保 `eval_loss` 存在
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }
    
from transformers import TrainingArguments, Trainer, EarlyStoppingCallback
training_args = TrainingArguments(
    output_dir="test_trainer-sft-mix",
    eval_strategy="epoch",
    save_strategy="epoch",  # 每个 epoch 保存一次模型
    save_steps=None,  # 取消按steps保存
    learning_rate=2e-5,  # 学习率  2e-5
    per_device_train_batch_size=64,  # 适当增加 batch_size，默认 8
    per_device_eval_batch_size=64,
    num_train_epochs=20,  # 降低训练轮数，避免过拟合
    weight_decay=0.02,  # 加入 L2 正则化
    load_best_model_at_end=True,  # 训练结束后加载最佳模型
    logging_strategy="epoch",  # 确保每个 epoch 打印 loss
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]  # 设置早停机制
)

trainer.train()