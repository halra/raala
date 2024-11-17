import torch
from flair.data import Sentence
from scipy.stats import entropy
from scipy.spatial.distance import jensenshannon
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Any


class ModelEvaluator:
    def __init__(self):
        self.name: str = ''

    @staticmethod
    def normalize(scores: List[float]) -> List[float]:
        total = sum(scores)
        return [score / total for score in scores] if total > 0 else [0.0 for _ in scores]

    @staticmethod
    def calculate_jsd(distributions: np.ndarray) -> float:
        mean_distribution = np.mean(distributions, axis=0)
        jsd_values = [
            jensenshannon(dist, mean_distribution, base=2) ** 2 for dist in distributions
        ]
        return np.mean(jsd_values)

    def _eval_predictions(
        self, models: List[Any], text: str, num_samples: int = 1
    ) -> Tuple[np.ndarray, np.ndarray]:
        predictions = []
        for _ in range(num_samples):
            sentence = Sentence(text)
            for model in models:
                model.classifier.predict(sentence, return_probabilities_for_all_classes=True)
                prediction = [label.score for label in sentence.labels]
                predictions.append(prediction)
        predictions_array = np.array(predictions)
        return predictions_array, predictions_array.mean(axis=0)

    def evaluate(
        self,
        workload: Any,
        eval_type: str = "ensemble",
        num_samples: int = 20,
        debug: bool = False,
    ) -> None:
        print(f"Starting {eval_type} evaluation...")
        if eval_type == "ensemble":
            models = workload.models
            num_samples = 1
        elif eval_type == "mc_dropout":
            model = workload.models[0]
            model.classifier.train()
            models = [model]
        else:
            raise ValueError(f"Unknown evaluation type: {eval_type}")

        labels = workload.labels
        df = workload.df.copy()
        metrics = {
            'tag': [], 'score': [], 'variance_all': [], 'variance_tag': [],
            'entropy_all': [], 'entropy_tag': [], 'uncertainty': [],
            'entropy_mean_prediction': [], 'mean_entropy_individual': [],
            'mutual_information': [], 'mean_variance': [], 'mean_jsd': []
        }
        label_probabilities = {label: [] for label in labels}

        for idx, row in df.iterrows():
            predictions_array, mean_prediction = self._eval_predictions(
                models, row["text"], num_samples
            )

            predictions_array /= predictions_array.sum(axis=1, keepdims=True)
            mean_prediction /= mean_prediction.sum()

            #MV

            avg_scores = predictions_array.mean(axis=0)
            majority_vote_idx = np.argmax(avg_scores) -1 # consider the <unk> label -> <unk>, label1, label2 -> removes the vote for <unk>
            majority_vote_tag = labels[majority_vote_idx]
            majority_vote_scores = predictions_array[:, majority_vote_idx]

            #metrics 
            combined_score = avg_scores[majority_vote_idx]
            uncertainty = np.std(majority_vote_scores)
            variance_tag = np.var(majority_vote_scores)
            variance_all = np.var(predictions_array)
            entropy_all = entropy(avg_scores, base=2)
            entropy_tag = entropy([combined_score, 1 - combined_score], base=2)

            #entropy
            entropies_individual = [entropy(pred, base=2) for pred in predictions_array]
            mean_entropy_individual = np.mean(entropies_individual)
            mutual_information = entropy_all - mean_entropy_individual

            #JSD
            mean_variance = np.mean(np.var(predictions_array, axis=0))
            mean_jsd = self.calculate_jsd(predictions_array)

            metrics['tag'].append(majority_vote_tag)
            metrics['score'].append(combined_score)
            metrics['variance_all'].append(variance_all)
            metrics['variance_tag'].append(variance_tag)
            metrics['entropy_all'].append(entropy_all)
            metrics['entropy_tag'].append(entropy_tag)
            metrics['uncertainty'].append(uncertainty)
            metrics['entropy_mean_prediction'].append(entropy_all)
            metrics['mean_entropy_individual'].append(mean_entropy_individual)
            metrics['mutual_information'].append(mutual_information)
            metrics['mean_variance'].append(mean_variance)
            metrics['mean_jsd'].append(mean_jsd)

            for i, label in enumerate(labels):
                label_probabilities[label].append(mean_prediction[i])

            if debug and idx % 100 == 0:
                print(f"Processed {idx + 1}/{len(df)} rows.")

        for key, values in metrics.items():
            df[key] = values
        for label, probs in label_probabilities.items():
            df[f"{label}_probability_model"] = probs

        workload.df = df
        print("Evaluation completed.")
