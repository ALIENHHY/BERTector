import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments, EarlyStoppingCallback
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

# 数据加载
df = pd.read_csv('NSL-KDD-Train-1000.csv')

# 模型路径
model_path = 'bert-base-uncased'
lr = 2e-5
batch_size = 16
epochs = 10
max_len = 512

# 加载tokenizer
tokenizer = BertTokenizer.from_pretrained(model_path)

# 计算准确率的函数
def compute_accuracy(p):
    predictions, labels = p
    preds = predictions.argmax(axis=-1)  # 获取预测结果的索引
    return {"accuracy": accuracy_score(labels, preds)}  # 返回字典，键为 'accuracy'

class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        text = str(self.texts[item])
        label = int(self.labels[item])  # 确保 label 是整数

        # 自定义分词：以逗号分隔，并忽略逗号
        words = text.split(',')  # 按逗号分割
        words = [word.strip() for word in words if word.strip()]  # 去除空格和空字符串

        # 进一步用 BERT tokenizer 进行子词分割
        encoding = tokenizer.encode_plus(
            words,  # 传入分词后的列表
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            is_split_into_words=True,  # 让 tokenizer 识别传入的是已经分好的词
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# 分割训练集和验证集
train_texts, val_texts, train_labels, val_labels = train_test_split(df['flow'], df['class'], test_size=0.2, random_state=42)

# 创建训练集和验证集
train_dataset = TextDataset(train_texts.tolist(), train_labels.tolist(), tokenizer, max_len)
val_dataset = TextDataset(val_texts.tolist(), val_labels.tolist(), tokenizer, max_len)

# 创建数据加载器
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)

# 模型和优化器
model = BertForSequenceClassification.from_pretrained(model_path, num_labels=2)

# 设置训练参数
training_args = TrainingArguments(
    output_dir="results",
    learning_rate=lr,
    num_train_epochs=epochs,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    evaluation_strategy="epoch",  # 每个epoch进行评估
    logging_dir='logs',
    logging_steps=10,
    weight_decay=0.01,
    save_strategy="epoch",  # 按epoch保存模型
    load_best_model_at_end=True,  # 加载最好的模型
)

# 训练器
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_accuracy,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]  # 设置早停机制
)

# 开始训练
trainer.train()
metrics = trainer.evaluate()