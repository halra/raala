import os
import random
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from flair.data import Corpus
from flair.datasets import CSVClassificationCorpus
from flair.embeddings import TransformerDocumentEmbeddings
from flair.models import TextClassifier
from flair.trainers import ModelTrainer
import flair
import logging
from typing import List, Optional

logger = logging.getLogger(__name__)

class ModelUtils:
    def __init__(
        self,
        instance_name,
        model_name,
        labels_array,
        work_dir=None,
        data_file='df.csv',
        column_name_map=None,
    ):
        self.instance_name = instance_name
        self.labels_array = labels_array
        self.gold_label_type = "label_gold"
        self.model_name = model_name
        self.work_dir = work_dir or os.path.join(os.getcwd(), "workload")
        self.data_file = data_file
        self.column_name_map = column_name_map or {0: "text", 3: self.gold_label_type}
        self.df = pd.read_csv(os.path.join(self.work_dir, self.data_file))
        self.corpus = self._create_corpus()
        self.label_dict = self.corpus.make_label_dictionary(label_type=self.gold_label_type)
        self.classifier = self._initialize_classifier()

    def _create_corpus(self) -> Corpus:
        return CSVClassificationCorpus(
            self.work_dir,
            self.column_name_map,
            skip_header=True,
            delimiter=',',
            label_type=self.gold_label_type,
        )

    def _initialize_classifier(self, dropout: float = 0.0) -> TextClassifier:
        embeddings = TransformerDocumentEmbeddings(
            self.model_name, fine_tune=True, dropout=dropout
        )
        return TextClassifier(
            embeddings,
            label_dictionary=self.label_dict,
            label_type=self.gold_label_type,
        )

    def load_model(self, model_path: str):
        model_path = os.path.join(model_path, "final-model.pt")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}")
        logger.info(f"Loading model from {model_path}")
        self.classifier = TextClassifier.load(model_path)

    def train(
        self,
        learning_rate: float = 5e-5,
        mini_batch_size: int = 8,
        max_epochs: int = 2,
        seed: int = 0,
        smoothing: float = 0.0,
    ):
        self._prepare_for_training(seed, smoothing)
        model_dir = self._get_model_directory()
        trainer = ModelTrainer(self.classifier, self.corpus)
        logger.info(
            f"Training model '{self.model_name}' with learning_rate={learning_rate}, "
            f"mini_batch_size={mini_batch_size}, max_epochs={max_epochs}, seed={seed}."
        )
        trainer.fine_tune(
            model_dir,
            learning_rate=learning_rate,
            mini_batch_size=mini_batch_size,
            max_epochs=max_epochs,
            monitor_test=True,
        )
        logger.info(f"Training completed. Model saved to '{model_dir}'.")

    def _prepare_for_training(self, seed: int, smoothing: float):
        self.set_seed(seed)
        if smoothing > 0.0:
            self.classifier.loss_function = nn.CrossEntropyLoss(label_smoothing=smoothing)
            logger.info(f"Using label smoothing: {smoothing}")

    def _get_model_directory(self) -> str:
        model_dir = os.path.join(os.getcwd(), "models", self.instance_name)
        os.makedirs(model_dir, exist_ok=True)
        return model_dir

    @staticmethod
    def set_seed(seed_value: int):
        random.seed(seed_value)
        np.random.seed(seed_value)
        torch.manual_seed(seed_value)
        flair.set_seed(seed_value)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed_value)

    def evaluate(self, corpus: Optional[Corpus] = None):
        corpus = corpus or self.corpus
        logger.info(f"Evaluating model '{self.model_name}'...")
        results = self.classifier.evaluate(corpus, gold_label_type=self.gold_label_type)
        logger.info(results.detailed_results)

    def change_df(self, path: str):
        if not os.path.exists(path):
            raise FileNotFoundError(f"DataFrame file not found at {path}")
        self.df = pd.read_csv(path)
        logger.info(f"DataFrame loaded from {path}")
        self.corpus = self._create_corpus()
        self.label_dict = self.corpus.make_label_dictionary(label_type=self.gold_label_type)

    def train_with_smoothing(
        self,
        learning_rate: float = 5e-5,
        mini_batch_size: int = 8,
        max_epochs: int = 2,
        seed: int = 0,
        smoothing: float = 0.2,
    ):
        self.train(learning_rate, mini_batch_size, max_epochs, seed, smoothing)

    def train_with_dropout(
        self,
        learning_rate: float = 5e-5,
        mini_batch_size: int = 8,
        max_epochs: int = 2,
        seed: int = 0,
        dropout: float = 0.5,
        smoothing: float = 0.0,
    ):
        logger.info(
            f"Training model '{self.model_name}' with dropout={dropout}, learning_rate={learning_rate}."
        )
        self.classifier = self._initialize_classifier(dropout=dropout)
        self.train(learning_rate, mini_batch_size, max_epochs, seed, smoothing)
