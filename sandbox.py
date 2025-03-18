import torch
import torch.nn as nn
import torch.nn.functional as F

class LabelSmoothingCrossEntropyLoss(nn.Module):
    def __init__(self, num_classes, reduction="mean"):

        super(LabelSmoothingCrossEntropyLoss, self).__init__()
        self.num_classes = num_classes
        self.reduction = reduction

    def forward(self, logits, target, smoothing):
        log_probs = F.log_softmax(logits, dim=-1)  
        
        one_hot = torch.zeros_like(log_probs)
        one_hot.scatter_(1, target.unsqueeze(1), 1.0)
        
        smoothing = smoothing.unsqueeze(1)
        
        smooth_labels = one_hot * (1.0 - smoothing) + (1.0 - one_hot) * (smoothing / (self.num_classes - 1))
        
        loss = -torch.sum(smooth_labels * log_probs, dim=-1)
        
        # Is this needed ?
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss


from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_length):
        self.dataframe = dataframe
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # mapping from string labels to integers
        self.label2id = {"hate": 1, "no_hate": 0}

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        text = row["text"]
        label_str = row["gold_label"].strip().lower()  
        label = self.label2id.get(label_str, 0)  # 0 if label is not found
        smoothing = row["highest_agreement"]

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt"
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),         
            "attention_mask": encoding["attention_mask"].squeeze(0), 
            "label": torch.tensor(label, dtype=torch.long),
            "smoothing": torch.tensor(smoothing, dtype=torch.float)
        }
from torch.utils.data import DataLoader
from transformers import BertForSequenceClassification, BertTokenizer
import pandas as pd

print("Loading data...")
df = pd.read_csv("./workload/train.csv")

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
dataset = CustomDataset(dataframe=df, tokenizer=tokenizer, max_length=128)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=3)
loss_fn = LabelSmoothingCrossEntropyLoss(num_classes=3)

optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)

model.train()
print("Starting training...")
for epoch in range(3):
    for batch in dataloader:
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["label"]
        smoothing = batch["smoothing"]  # this is what we want!
        #print("batch", batch)
        print("smoothing", smoothing)
        print("batch['label']", batch["label"])
        #print("batch['text']", batch["text"])
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits

        loss = loss_fn(logits, labels, smoothing)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        print(f"Loss: {loss.item()}")
