import os
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from transformers import BertForSequenceClassification, BertTokenizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

#logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
MODEL_PATH = "./models/test"
NUM_EPOCHS = 10
BATCH_SIZE = 8
MODEL_NAME = "bert-base-uncased"

# -----------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------
def label_2_id_processor(df):
    label_columns = [col for col in df.columns if col.endswith("_probability")]
    labels = [col.replace("_probability", "") for col in label_columns]
    labels = sorted(labels)  # Sorting for consistent ordering
    LABEL_2_ID = {label: idx for idx, label in enumerate(labels)}
    logger.info(f"Generated LABEL_2_ID mapping: {LABEL_2_ID}")
    return LABEL_2_ID

# -----------------------------------------------------------------------------
# Custom Loss with Label Smoothing
# -----------------------------------------------------------------------------
class LabelSmoothingCrossEntropyLoss(nn.Module):
    def __init__(self, num_classes, reduction="mean"):
        super(LabelSmoothingCrossEntropyLoss, self).__init__()
        self.num_classes = num_classes
        self.reduction = reduction

    def forward(self, logits, soft_labels):
        #print("soft_labels", soft_labels[0]) # just look at the first element form the batch
        loss = nn.CrossEntropyLoss()
        output = loss(logits, soft_labels)
        if True:
            return output
        log_probs = F.log_softmax(logits, dim=-1)
        loss = -(soft_labels * log_probs).sum(dim=-1)
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss

# -----------------------------------------------------------------------------
# Custom Dataset
# -----------------------------------------------------------------------------
class CustomDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_length, num_classes):
        self.dataframe = dataframe
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.num_classes = num_classes

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        text = row["text"]
        label_str = row["gold_label"].strip().lower()
        # Use global LABEL_2_ID mapping (processed later)
        gold_class = LABEL_2_ID.get(label_str, 0)
        highest_agreement = row["highest_agreement"]

        # gold_label gets highest_agreement and others share the remainder.
        soft_label = [(1 - highest_agreement) / (self.num_classes - 1)] * self.num_classes
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

# -----------------------------------------------------------------------------
# Main Training and Evaluation Function
# -----------------------------------------------------------------------------
def train_and_evaluate():
    logger.info("Loading data...")
    df = pd.read_csv("./workload/train.csv") # TODO lead train and validate on dev and test .... 
    df_dev = pd.read_csv("./workload/dev.csv")
    global LABEL_2_ID
    LABEL_2_ID = label_2_id_processor(df)
    num_classes = len(LABEL_2_ID)
    logger.info(f"Number of classes: {num_classes}")
    
    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
    dataset = CustomDataset(dataframe=df, tokenizer=tokenizer, max_length=128, num_classes=num_classes)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    dataset_dev = CustomDataset(dataframe=df_dev, tokenizer=tokenizer, max_length=128, num_classes=num_classes)
    dataloader_dev = DataLoader(dataset_dev, batch_size=BATCH_SIZE, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    model = BertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=num_classes)
    model.to(device)

    loss_fn = LabelSmoothingCrossEntropyLoss(num_classes=num_classes)
    loss_fn.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)

    logger.info("Starting training with soft labels...")
    for epoch in range(NUM_EPOCHS):
        epoch_losses = []
        model.train()
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            soft_labels = batch["soft_label"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            loss = loss_fn(logits, soft_labels)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            epoch_losses.append(loss.item())

        avg_loss = sum(epoch_losses) / len(epoch_losses)
        logger.info(f"Epoch {epoch+1}/{NUM_EPOCHS} - Average Training Loss: {avg_loss:.4f}")

        # Evaluation
        model.eval()
        all_preds = []
        all_true = []
        with torch.no_grad():
            for batch in dataloader_dev:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                soft_labels = batch["soft_label"].to(device)

                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                preds = torch.argmax(logits, dim=1)
                true_labels = torch.argmax(soft_labels, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_true.extend(true_labels.cpu().numpy())

        acc = accuracy_score(all_true, all_preds)
        prec = precision_score(all_true, all_preds, average='weighted', zero_division=0)
        rec = recall_score(all_true, all_preds, average='weighted', zero_division=0)
        f1 = f1_score(all_true, all_preds, average='weighted', zero_division=0)
        logger.info(f"Epoch {epoch+1} Evaluation:")
        logger.info(f"  Accuracy: {acc:.4f}")
        logger.info(f"  Precision: {prec:.4f}")
        logger.info(f"  Recall: {rec:.4f}")
        logger.info(f"  F1 Score: {f1:.4f}")
        logger.info("Classification Report:")
        logger.info("\n" + classification_report(all_true, all_preds, zero_division=0))
        logger.info("-" * 60)

    # -----------------------------------------------------------------------------
    # Save the model in a Flair-compatible format
    # -----------------------------------------------------------------------------
    logger.info("Finished training, saving model using Flair format...")
    from flair.embeddings import TransformerDocumentEmbeddings
    from flair.models import TextClassifier
    from flair.data import Dictionary

    label_dict = Dictionary(add_unk=False)
    for label in sorted(LABEL_2_ID.keys()):
        label_dict.add_item(label)
    document_embeddings = TransformerDocumentEmbeddings(MODEL_NAME) 
    flair_classifier = TextClassifier(document_embeddings, label_dictionary=label_dict, label_type="label_gold")
    missing_keys = flair_classifier.load_state_dict(model.state_dict(), strict=False)
    #logger.info(f"Missing keys when loading state dict into Flair classifier: {missing_keys}")
    flair_save_path = os.path.join(MODEL_PATH, "final-model.pt")
    flair_classifier.save(flair_save_path)
    logger.info(f"Flair model saved at: {flair_save_path}")

    return model, flair_classifier, tokenizer, device, LABEL_2_ID

# -----------------------------------------------------------------------------
# Inference Function Using Flair
# -----------------------------------------------------------------------------
def run_inference(flair_classifier, df):
    from flair.data import Sentence
    logger.info("Running inference on dataset...")
    for index, row in df.iterrows():
        text = row["text"]
        highest_agreement = row["highest_agreement"]
        sentence = Sentence(text)
        #flair_classifier.predict(sentence, return_probabilities_for_all_classes=True)
        flair_classifier.predict(sentence, return_probabilities_for_all_classes=False)
        pred_label = sentence.labels[0].value
        pred_prob = sentence.labels[0].score
        logger.info(f"Text: {sentence.text}")
        logger.info(f"Predicted Label: {pred_label} with probability: {pred_prob:.4f} and highest agreement: {highest_agreement:.4f}")
        logger.info("-" * 40)

# -----------------------------------------------------------------------------
# Main Entry Point
# -----------------------------------------------------------------------------
from flair.models import TextClassifier
if __name__ == "__main__":
    trained_model, flair_cl, tokenizer, device, label2id = train_and_evaluate()
    df_test = pd.read_csv("./workload/test.csv")
    cl = TextClassifier.load(MODEL_PATH + "/final-model.pt")
    run_inference(cl, df_test)
