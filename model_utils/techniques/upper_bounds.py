import os
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import random
import numpy as np
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)

# Constants
#MODEL_PATH = "./models/test"
#NUM_EPOCHS = 10
#BATCH_SIZE = 8
#MODEL_NAME = "bert-base-uncased"

#logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


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
        return output


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
        return self._process_data_point(row)
    
    def _process_data_point(self, row):
        text = row["text"]
        label_str = row["gold_label"].strip().lower()
        # Use global LABEL_2_ID mapping (processed later)
        gold_class = LABEL_2_ID.get(label_str, 0)
        highest_agreement = row["highest_agreement"]

        # each label gets the corresponding empirical probability, <unk> is 0
        soft_label = [0] * self.num_classes
        for label_type in LABEL_2_ID:    
            if label_type != '<unk>':
                soft_label[LABEL_2_ID[label_type]] = row[label_type + "_probability"]
        
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




class Upper_bounds_trainer():
    def __init__(
        self,
        instance_name,
        model_name,
    ):
        self.model_name = model_name
        self.instance_name = instance_name
        self.model_path = os.path.join(os.getcwd(), "models", self.instance_name)
        label_dictionary = None
        num_classes = 0
        device = 'cpu'
        model = None
        tokenizer = None


    #logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

    # Constants
    #MODEL_PATH = "./models/test"
    NUM_EPOCHS = 10
    BATCH_SIZE = 8
    #MODEL_NAME = "bert-base-uncased"

    # -----------------------------------------------------------------------------
    # Helper Functions
    # -----------------------------------------------------------------------------
    def _label_2_id_processor(self, df):
        label_columns = [col for col in df.columns if col.endswith("_probability")]
        labels = [col.replace("_probability", "") for col in label_columns]
        labels.append("<unk>")  # Add 'other' class for non-annotated data
        labels = sorted(labels)  # Sorting for consistent ordering
        LABEL_2_ID = {label: idx for idx, label in enumerate(labels)}
        logger.info(f"Generated LABEL_2_ID mapping: {LABEL_2_ID}")
        return LABEL_2_ID
    
    @staticmethod
    def set_seed(seed_value: int):
        random.seed(seed_value)
        np.random.seed(seed_value)
        torch.manual_seed(seed_value)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed_value)


    # -----------------------------------------------------------------------------
    # Main Training and Evaluation Function
    # -----------------------------------------------------------------------------
    def train(
        self,
        learning_rate: float = 5e-5,
        mini_batch_size: int = 8,
        max_epochs: int = 2,
        seed: int = 0,
        smoothing: float = 0.0,
    ):
        
        self.set_seed(seed)
        logger.info("Loading data...")
        df = pd.read_csv("./workload/train.csv") # TODO lead train and validate on dev and test .... 
        df_dev = pd.read_csv("./workload/dev.csv")
        global LABEL_2_ID
        LABEL_2_ID = self._label_2_id_processor(df)
        self.label_dictionary = LABEL_2_ID
        self.num_classes = len(LABEL_2_ID)
        logger.info(f"Number of classes: {self.num_classes}")
        
        #tokenizer = BertTokenizer.from_pretrained(self.model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
        )
        dataset = CustomDataset(dataframe=df, tokenizer=self.tokenizer, max_length=128, num_classes=self.num_classes)
        dataloader = DataLoader(dataset, batch_size=mini_batch_size, shuffle=True)
        dataset_dev = CustomDataset(dataframe=df_dev, tokenizer=self.tokenizer, max_length=128, num_classes=self.num_classes)
        dataloader_dev = DataLoader(dataset_dev, batch_size=mini_batch_size, shuffle=True)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")

        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name, num_labels=self.num_classes)
        
        # TODO if model is None raise panic
        self.model.to(self.device)

        loss_fn = LabelSmoothingCrossEntropyLoss(num_classes=self.num_classes)
        loss_fn.to(self.device)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate) # TODO this should be a parameter and not hardcoed AdamW

        logger.info("Starting training with soft labels...")
        for epoch in range(max_epochs):
            epoch_losses = []
            self.model.train()
            for batch in dataloader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                soft_labels = batch["soft_label"].to(self.device)

                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                loss = loss_fn(logits, soft_labels)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                epoch_losses.append(loss.item())

            avg_loss = sum(epoch_losses) / len(epoch_losses)
            logger.info(f"Epoch {epoch+1}/{max_epochs} - Average Training Loss: {avg_loss:.4f}")

            # Evaluation
            self.model.eval()
            all_preds = []
            all_true = []
            with torch.no_grad():
                for batch in dataloader_dev:
                    input_ids = batch["input_ids"].to(self.device)
                    attention_mask = batch["attention_mask"].to(self.device)
                    soft_labels = batch["soft_label"].to(self.device)

                    outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
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
        document_embeddings = TransformerDocumentEmbeddings(self.model_name) 
        flair_classifier = TextClassifier(document_embeddings, label_dictionary=label_dict, label_type="label_gold")
        missing_keys = flair_classifier.load_state_dict(self.model.state_dict(), strict=False)
        #logger.info(f"Missing keys when loading state dict into Flair classifier: {missing_keys}")
        flair_save_path = os.path.join(self.model_path, "final-model.pt")
        os.makedirs(self.model_path, exist_ok=True)
        flair_classifier.save(flair_save_path)
        logger.info(f"Flair model saved at: {flair_save_path}")
        logger.info(f"Training completed. Model saved to '{flair_save_path}'.")
        return self.model, flair_classifier, self.tokenizer, self.device, LABEL_2_ID

    def predict(
            self, row
        ):
        datapoint_dataset = CustomDataset(dataframe=pd.DataFrame([row]), tokenizer=self.tokenizer, max_length=128, num_classes=self.num_classes)    
        datapoint_dataloader = DataLoader(datapoint_dataset, batch_size=1, shuffle=True)
        # define iterator for use in training
        iterator = iter(datapoint_dataloader)

        # extract batch
        data_point = next(iterator)

        input_ids = data_point["input_ids"].to(self.device)
        attention_mask = data_point["attention_mask"].to(self.device)
        soft_labels = data_point["soft_label"].to(self.device)
        self.model.eval()

        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            softmax = torch.softmax(outputs.logits, dim=1)
            softmax = softmax.cpu().numpy()[0]

        LABEL_2_ID_REVERSED = {v: k for k, v in LABEL_2_ID.items()}

        return [LABEL_2_ID_REVERSED[idx] for idx in range(len(softmax))], softmax.tolist()