import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
def label_2_id_processor(df):
    label_columns = [col for col in df.columns if col.endswith("_probability")]
    labels = [col.replace("_probability", "") for col in label_columns]
    labels = sorted(labels) # just cause we can
    LABEL_2_ID = {label: idx for idx, label in enumerate(labels)}
    print(LABEL_2_ID)
    return LABEL_2_ID


model_path = "./models/test"
NUM_CLASSES = 28
NUM_EPOCHS = 6
BATCH_SIZE = 8
# this is an example, a placeholder for the actual mapping
LABEL_2_ID = {
    "admiration": 0,
    "amusement": 1,
    "...": 9999
    }

class LabelSmoothingCrossEntropyLoss(nn.Module):
    def __init__(self, num_classes, reduction="mean"):

        super(LabelSmoothingCrossEntropyLoss, self).__init__()
        self.num_classes = num_classes
        self.reduction = reduction

    def forward(self, logits, smoothing):
        
        loss = nn.CrossEntropyLoss()
        output = loss(logits, smoothing)
        if True:
            return output
        # this is not needed when we use CrossEntropyLoss
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


    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        text = row["text"]
        label_str = row["gold_label"].strip().lower()
        gold_class = LABEL_2_ID.get(label_str, 0)
        #print("gold_class", gold_class, "label_str", label_str)
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
            "soft_label": soft_label,
            "debug_text": text
        }
        
from torch.utils.data import DataLoader
from transformers import BertForSequenceClassification, BertTokenizer
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report


print("Loading data...")
df = pd.read_csv("./workload/train.csv")
LABEL_2_ID = label_2_id_processor(df)
NUM_CLASSES = len(LABEL_2_ID)

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

dataset = CustomDataset(dataframe=df, tokenizer=tokenizer, max_length=128, num_classes=NUM_CLASSES)

dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True) # use batch size 1 to better debug


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=NUM_CLASSES)
model.to(device)


loss_fn = LabelSmoothingCrossEntropyLoss(num_classes=NUM_CLASSES)
loss_fn.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)




print("Starting training with soft labels...")
for epoch in range(NUM_EPOCHS):
    epoch_losses = []
    model.train()
    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        soft_labels = batch["soft_label"].to(device)
        #print("soft_labels", soft_labels[0]) # only peek at first soft label
        #print("debug_text", batch["debug_text"][0]) # peek at first text

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits  
        
        loss = loss_fn(logits, soft_labels)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        epoch_losses.append(loss.item())
        #print(f"Epoch {epoch+1}, Loss: {loss.item()}")
        
    # Evaluation
    model.eval()
    all_preds = []
    all_true = []
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            soft_labels = batch["soft_label"].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            
            # predicted classes
            preds = torch.argmax(logits, dim=1)
            # gold class has highest probability
            true_labels = torch.argmax(soft_labels, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_true.extend(true_labels.cpu().numpy())
    
    acc = accuracy_score(all_true, all_preds)
    prec = precision_score(all_true, all_preds, average='weighted', zero_division=0)
    rec = recall_score(all_true, all_preds, average='weighted', zero_division=0)
    f1 = f1_score(all_true, all_preds, average='weighted', zero_division=0)
    
    print(f"\nEpoch {epoch+1} Evaluation:")
    print(f"Average Training Loss: {sum(epoch_losses)/len(epoch_losses):.4f}")
    print(f"Accuracy: {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall: {rec:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print("Classification Report:")
    print(classification_report(all_true, all_preds, zero_division=0))
    print("-" * 60)
        
        
print("Finished training, saving model...")



#https://pytorch.org/tutorials/beginner/saving_loading_models.html

from flair.embeddings import TransformerDocumentEmbeddings
if False:
    #DEBUG
    from flair.embeddings import TransformerDocumentEmbeddings
    from flair.models import TextClassifier
    from flair.data import Dictionary

    document_embeddings = TransformerDocumentEmbeddings('bert-base-uncased')
    label_dict = Dictionary(add_unk=False)
    for label in ["hate", "no_hate"]:
        label_dict.add_item(label)
    flair_classifier = TextClassifier(document_embeddings, label_dictionary=label_dict, label_type="label_gold")

    # Load the classifier state dict into flair_classifier.classifier.
    #flair_classifier.classifier.load_state_dict(model.state_dict())
    missing_keys = flair_classifier.load_state_dict(model.state_dict(), strict=False)
    flair_classifier.save(model_path)    
    #END DEBUG


#torch.save(model.state_dict(), model_path)




from flair.models import TextClassifier
from flair.data import Sentence

#classifier = TextClassifier.load(model_path)
#sentence = Sentence("Sum random text")
#model.classifier.predict(sentence, return_probabilities_for_all_classes=True)
#print(sentence.labels)
#print(sentence.labels[0].score)



#DEBUG 2

def predict_text(text, model, tokenizer, device, id2label, max_length=128):
    encoding = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=max_length,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt'
    )
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    
    # Inference
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        probabilities = torch.softmax(logits, dim=-1)
        predicted_class = torch.argmax(probabilities, dim=-1)
    
    #human-readable label
    predicted_id = predicted_class.item()
    predicted_label = id2label.get(predicted_id)
    
    #print("Text:", text)
    #print("Predicted Label:", predicted_label)
    probs = probabilities.cpu().numpy()[0]  # Assuming batch size 1
    predicted_probability = probabilities.cpu().numpy()[0][predicted_id]
    #print("Probabilities per label:")
    for idx, prob in enumerate(probs):
        label_name = id2label.get(idx)
        #print(f"  {label_name}: {prob:.4f}")
    #print("-" * 40)
    
    return predicted_label, predicted_probability
    
# get label from integer 
id2label = {v: k for k, v in LABEL_2_ID.items()}
model.eval()   
    
#for index, row in df.iterrows():
#    text = row["text"]
#    predicted_label, predicted_probability = predict_text(text, model, tokenizer, device, id2label)
#    print(f"Text: {text}")
#    print(f"Predicted Label: {predicted_label} with probability: {predicted_probability:.4f}")
#    print("-" * 40)
    
    
#torch.save({
#    'state_dict': model.state_dict(),
#    'num_labels': NUM_CLASSES,
#    'bert_model_path': 'finetuned_bert'
#}, model_path)


#https://github.com/flairNLP/flair/issues/2072
import os
model_path = os.path.join(os.getcwd(), "models", "test")
os.makedirs(model_path, exist_ok=True)

#model.save_pretrained(model_path)
model_path = os.path.join(model_path, "final-model.pt")

document_embeddings = TransformerDocumentEmbeddings('bert-base-uncased')
document_embeddings.state_dict()

#DEBUG3 Save
#https://discuss.pytorch.org/t/missing-keys-unexpected-keys-in-state-dict-when-loading-self-trained-model/22379/13
#https://flairnlp.github.io/flair/v0.13.1/api/flair.models.html#flair.models.TextClassifier
torch.save({
                'epoch': NUM_EPOCHS,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'state_dict': model.state_dict(),
                #'loss_histories': loss_histories,
                "embeddings": document_embeddings.state_dict(),
                }, model_path)


#END DEBUG 3 Save



print("Finished saving model")


TextClassifier.load(model_path)