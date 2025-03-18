import torch
import torch.nn as nn
import torch.nn.functional as F

class LabelSmoothingCrossEntropyLoss(nn.Module):
    def __init__(self, num_classes, reduction="mean"):

        super(LabelSmoothingCrossEntropyLoss, self).__init__()
        self.num_classes = num_classes
        self.reduction = reduction

    def forward(self, logits, smoothing):
        log_probs = F.log_softmax(logits, dim=-1)
        loss = -(smoothing * log_probs).sum(dim=-1)
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss


from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_length, num_classes):
        self.dataframe = dataframe
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.num_classes = num_classes
        
        # mapping from string labels to integers
        self.label2id = {"hate": 1, "no_hate": 0}

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        text = row["text"]
        label_str = row["gold_label"].strip().lower()
        gold_class = self.label2id.get(label_str, 0) 
        highest_agreement = row["highest_agreement"] 
        
        
        # this assumes that only the gold label is unique and the rest is disributed equally ...  tho that is ok, since we onyl eval. on the gold_label
        soft_label = [ (1 - highest_agreement) / (self.num_classes - 1) ] * self.num_classes
        soft_label[gold_class] = highest_agreement
        soft_label = torch.tensor(soft_label, dtype=torch.float)
        
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
            "soft_label": soft_label
        }
        
from torch.utils.data import DataLoader
from transformers import BertForSequenceClassification, BertTokenizer
import pandas as pd

print("Loading data...")
df = pd.read_csv("./workload/train.csv")

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

dataset = CustomDataset(dataframe=df, tokenizer=tokenizer, max_length=128, num_classes=2)

dataloader = DataLoader(dataset, batch_size=8, shuffle=True) # use batch size 1 to better debug

model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
loss_fn = LabelSmoothingCrossEntropyLoss(num_classes=3)

optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)

model.train()
print("Starting training with soft labels...")
for epoch in range(2):
    for batch in dataloader:
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        soft_labels = batch["soft_label"] 
        print("soft_labels", soft_labels) # print #batch size labels

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits  
        
        loss = loss_fn(logits, soft_labels)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        print(f"Epoch {epoch+1}, Loss: {loss.item()}")