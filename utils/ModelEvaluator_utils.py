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

    def normalize(self, scores: List[float]) -> List[float]:

        total = sum(scores)
        if total > 0:
            return [float(score) / total for score in scores]
        else:
            return [0.0 for _ in scores]

    def calculate_jsd(self, distributions: List[List[float]]) -> float:
 
        average_distribution = np.mean(distributions, axis=0)
        jsd_values = [
            jensenshannon(dist, average_distribution, base=2) ** 2 for dist in distributions
        ]
        return np.mean(jsd_values)

    def _eval_deep_ensemble(
        self, workload: Any, row: pd.Series
    ) -> Tuple[List[Dict[str, float]], List[float]]:

        list_label_and_scores = []
        all_classes_scores = []
        
        for model in workload.models:
            sentence = Sentence(row["text"])
            model.classifier.predict(sentence, return_probabilities_for_all_classes=True)
            label_and_scores = {label.value: label.score for label in sentence.labels}
            list_label_and_scores.append(label_and_scores)
            all_classes_scores.extend([label.score for label in sentence.labels])
        
        return list_label_and_scores, all_classes_scores

    def _eval_mc_dropout(
        self, workload: Any, row: pd.Series, num_samples: int
    ) -> Tuple[List[Dict[str, float]], List[float]]:

        list_label_and_scores = []
        all_classes_scores = []
        
        for _ in range(num_samples):
            with torch.no_grad():
                sentence = Sentence(row["text"])
                workload.models[0].classifier.predict(sentence, return_probabilities_for_all_classes=True)
                label_and_scores = {label.value: label.score for label in sentence.labels}
                list_label_and_scores.append(label_and_scores)
                all_classes_scores.extend([label.score for label in sentence.labels])
        
        return list_label_and_scores, all_classes_scores

    def _calculate_majority_vote(
        self, list_label_and_scores: List[Dict[str, float]]
    ) -> Tuple[str, List[float]]:

        tag_scores = {}
        for prediction in list_label_and_scores:
            for tag, score in prediction.items():
                tag_scores[tag] = tag_scores.get(tag, 0.0) + score

        majority_vote_tag = max(tag_scores, key=tag_scores.get)
        majority_vote_scores = [
            prediction.get(majority_vote_tag, 0.0) for prediction in list_label_and_scores
        ]
        return majority_vote_tag, majority_vote_scores

    def _calculate_metrics(
        self,
        majority_vote_scores: List[float],
        all_classes_scores: List[float],
        normalizing_value: int,
    ) -> Tuple[float, float, float, float, float, float]:
        
        combined_score = sum(majority_vote_scores) / normalizing_value
        uncertainty = (
            torch.std(torch.tensor(majority_vote_scores)).item()
            if len(majority_vote_scores) > 1
            else 0.0
        )

        variance_tag = np.var(majority_vote_scores) if len(majority_vote_scores) > 1 else 0.0
        variance_all = np.var(all_classes_scores) if len(all_classes_scores) > 1 else 0.0

        normalized_all_scores = self.normalize(all_classes_scores)
        entropy_all = entropy(normalized_all_scores, base=2) if normalized_all_scores else 0.0

        normalized_tag_scores = self.normalize(majority_vote_scores)
        entropy_tag = entropy(normalized_tag_scores, base=2) if normalized_tag_scores else 0.0

        return combined_score, uncertainty, variance_tag, variance_all, entropy_all, entropy_tag



    def evaluate(
        self,
        workload: Any,
        eval_type: str = "ensemble",
        num_samples: int = 20,
        debug: bool = False,
    ) -> None:
        print(f"Starting evaluation with type: {eval_type}")
        print(
            f"Number of models: {len(workload.models)}, "
            f"Number of samples: {num_samples if eval_type == 'mc_dropout' else 'N/A'}"
        )
        
        normalizing_value = len(workload.models) if eval_type == "ensemble" else num_samples

        tags = []
        scores = []
        variances_all = []
        variances_tag = []
        entropies_all = []
        entropies_tag = []
        uncertainties = []
        true_labels = []

        entropy_mean_predictions = []
        mean_entropy_individuals = []
        mutual_informations = []
        mean_variances = []
        mean_jsds = []

        label_probabilities = {label: [] for label in workload.labels}

        total_rows = len(workload.df)
        for i, row in workload.df.iterrows():
            if eval_type == "ensemble":
                list_label_and_scores, all_classes_scores = self._eval_deep_ensemble(workload, row)
            elif eval_type == "mc_dropout":
                workload.models[0].classifier.train()
                list_label_and_scores, all_classes_scores = self._eval_mc_dropout(
                    workload, row, num_samples
                )
                workload.models[0].classifier.eval()
            else:
                raise ValueError(f"Unknown evaluation type: {eval_type}")
            
            if debug:
                percent_complete = (i + 1) / total_rows * 100
                print(f"\nProcessing row {i+1}/{total_rows} ({percent_complete:.2f}% complete)")
                print(f"Text: {row['text'][:50]}...")  # snippet of the text
                print(f"List of label and scores: {list_label_and_scores}")
                print(f"All class scores: {all_classes_scores}")

            majority_vote_tag, majority_vote_scores = self._calculate_majority_vote(
                list_label_and_scores
            )
            (
                combined_score,
                uncertainty,
                variance_tag,
                variance_all,
                entropy_all,
                entropy_tag,
            ) = self._calculate_metrics(
                majority_vote_scores, all_classes_scores, normalizing_value
            )
            
            predictions_array = np.array([
                [prediction.get(label, 0.0) for label in workload.labels]
                for prediction in list_label_and_scores
            ])

            predictions_array = predictions_array / predictions_array.sum(axis=1, keepdims=True)

            mean_prediction = predictions_array.mean(axis=0)

            for idx, label in enumerate(workload.labels):
                label_probabilities[label].append(mean_prediction[idx])

            entropy_mean, mean_entropy_individual, mutual_information = self.compute_entropy_metrics(predictions_array)

            variance_per_class, mean_variance = self.compute_variance(predictions_array)

            jsd_values, mean_jsd = self.compute_jsd(predictions_array)

            if debug:
                print(f"Majority vote tag: {majority_vote_tag}")
                print(f"Combined score: {combined_score}")
                print(f"Uncertainty: {uncertainty}")
                print(f"Variance (Tag): {variance_tag}, Variance (All): {variance_all}")
                print(f"Entropy (Tag): {entropy_tag}, Entropy (All): {entropy_all}")
                print(f"Entropy of Mean Prediction: {entropy_mean}")
                print(f"Mean Entropy of Individual Predictions: {mean_entropy_individual}")
                print(f"Mutual Information: {mutual_information}")
                print(f"Variance per Class: {variance_per_class}")
                print(f"Mean Variance: {mean_variance}")
                print(f"Mean Jensen-Shannon Divergence: {mean_jsd}")

            tags.append(majority_vote_tag)
            scores.append(combined_score)
            variances_all.append(variance_all)
            variances_tag.append(variance_tag)
            entropies_all.append(entropy_all)
            entropies_tag.append(entropy_tag)
            uncertainties.append(uncertainty)
            true_labels.append(row.get('gold_label', None))

            entropy_mean_predictions.append(entropy_mean)
            mean_entropy_individuals.append(mean_entropy_individual)
            mutual_informations.append(mutual_information)
            mean_variances.append(mean_variance)
            mean_jsds.append(mean_jsd)

        workload.df['tag'] = tags
        workload.df['score'] = scores
        workload.df['variance_all'] = variances_all
        workload.df['variance_tag'] = variances_tag
        workload.df['entropy_all'] = entropies_all
        workload.df['entropy_tag'] = entropies_tag
        workload.df['uncertainty'] = uncertainties
        workload.df['true_label'] = true_labels 

        workload.df['entropy_mean_prediction'] = entropy_mean_predictions
        workload.df['mean_entropy_individual'] = mean_entropy_individuals
        workload.df['mutual_information'] = mutual_informations
        workload.df['mean_variance'] = mean_variances
        workload.df['mean_jsd'] = mean_jsds

        for label in workload.labels:
            workload.df[f"{label}_probability_model"] = label_probabilities[label]

        print(f"Evaluation completed. Processed {total_rows} rows.")

    def compute_entropy_metrics(self, predictions_array):
        mean_prediction = np.mean(predictions_array, axis=0)
        entropy_mean = entropy(mean_prediction, base=2)
        entropies_individual = [entropy(pred, base=2) for pred in predictions_array]
        mean_entropy_individual = np.mean(entropies_individual)
        mutual_information = entropy_mean - mean_entropy_individual
        return entropy_mean, mean_entropy_individual, mutual_information

    def compute_variance(self, predictions_array):
        variance_per_class = np.var(predictions_array, axis=0)
        mean_variance = np.mean(variance_per_class)
        return variance_per_class, mean_variance

    def compute_jsd(self, predictions_array):
        mean_prediction = np.mean(predictions_array, axis=0)
        jsd_values = [
            jensenshannon(pred, mean_prediction, base=2)**2 for pred in predictions_array
        ]
        mean_jsd = np.mean(jsd_values)
        return jsd_values, mean_jsd
