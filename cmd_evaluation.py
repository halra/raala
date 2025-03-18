import os
import sys
import logging
import numpy as np
import pandas as pd
import torch
from typing import Any, List, Dict, Tuple
from scipy.stats import entropy
from scipy.spatial.distance import jensenshannon

from domain.workload import Workload
from flair.data import Sentence

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s %(levelname)s:%(message)s')
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)


def list_available_workloads(saved_results_dir='saved_results'):
    if not os.path.exists(saved_results_dir):
        logger.error(f"Saved results directory '{saved_results_dir}' does not exist.")
        sys.exit(1)
    workloads = [name for name in os.listdir(saved_results_dir)
                 if os.path.isdir(os.path.join(saved_results_dir, name))]
    return workloads


def _eval_deep_ensemble(workload: Any, text: str) -> Tuple[List[Dict[str, float]], List[float]]:
    list_label_and_scores = []
    all_classes_scores = []

    for model in workload.models:
        sentence = Sentence(text)
        model.classifier.predict(sentence, return_probabilities_for_all_classes=True)
        label_and_scores = {label.value: label.score for label in sentence.labels}
        list_label_and_scores.append(label_and_scores)
        all_classes_scores.extend([label.score for label in sentence.labels])

    return list_label_and_scores, all_classes_scores


def _eval_mc_dropout(workload: Any, text: str, num_samples: int) -> Tuple[List[Dict[str, float]], List[float]]:
    list_label_and_scores = []
    all_classes_scores = []

    workload.models[0].classifier.train()
    for _ in range(num_samples):
        with torch.no_grad():
            sentence = Sentence(text)
            workload.models[0].classifier.predict(sentence, return_probabilities_for_all_classes=True)
            label_and_scores = {label.value: label.score for label in sentence.labels}
            list_label_and_scores.append(label_and_scores)
            all_classes_scores.extend([label.score for label in sentence.labels])
    workload.models[0].classifier.eval()
    return list_label_and_scores, all_classes_scores


def _calculate_majority_vote(list_label_and_scores: List[Dict[str, float]]) -> Tuple[str, Dict[str, float]]:
    total_scores = {}
    for prediction in list_label_and_scores:
        for label, score in prediction.items():
            total_scores[label] = total_scores.get(label, 0.0) + score
    average_scores = {label: score / len(list_label_and_scores) for label, score in total_scores.items()}
    majority_vote_tag = max(average_scores, key=average_scores.get)
    return majority_vote_tag, average_scores


def _calculate_metrics(
    majority_vote_scores: Dict[str, float],
    all_classes_scores: List[float],
    normalizing_value: int,
) -> Tuple[float, float, float, float, float, float]:
    combined_score = sum(majority_vote_scores.values()) / len(majority_vote_scores)
    uncertainty = np.std(all_classes_scores)
    variance_all = np.var(all_classes_scores)
    variance_tag = np.var(list(majority_vote_scores.values()))
    entropy_all = entropy(all_classes_scores)
    entropy_tag = entropy(list(majority_vote_scores.values()))
    return combined_score, uncertainty, variance_tag, variance_all, entropy_all, entropy_tag


def compute_entropy_metrics(predictions_array):
    mean_prediction = predictions_array.mean(axis=0)
    entropy_mean_prediction = entropy(mean_prediction, base=2)
    entropies_individual = [entropy(pred, base=2) for pred in predictions_array]
    mean_entropy_individual = np.mean(entropies_individual)
    mutual_information = entropy_mean_prediction - mean_entropy_individual
    return entropy_mean_prediction, mean_entropy_individual, mutual_information


def compute_variance(predictions_array):
    variance_per_class = np.var(predictions_array, axis=0)
    mean_variance = np.mean(variance_per_class)
    return variance_per_class, mean_variance


def compute_jsd(predictions_array):
    mean_prediction = predictions_array.mean(axis=0)
    jsd_values = [jensenshannon(pred, mean_prediction, base=2) ** 2 for pred in predictions_array]
    mean_jsd = np.mean(jsd_values)
    return jsd_values, mean_jsd


def evaluate_text(workload, text, eval_type='ensemble', num_samples=100, debug=False):
    print(f"Starting evaluation with type: {eval_type}")
    labels = workload.labels  
    num_labels = len(labels)

    if eval_type == "ensemble":
        normalizing_value = len(workload.models)
        list_label_and_scores, all_classes_scores = _eval_deep_ensemble(workload, text)
    elif eval_type == "mc_dropout":
        normalizing_value = num_samples
        list_label_and_scores, all_classes_scores = _eval_mc_dropout(workload, text, num_samples)
    else:
        raise ValueError(f"Unknown evaluation type: {eval_type}")

    majority_vote_tag, majority_vote_scores = _calculate_majority_vote(list_label_and_scores)

    combined_score, uncertainty, variance_tag, variance_all, entropy_all, entropy_tag = _calculate_metrics(
        majority_vote_scores, all_classes_scores, normalizing_value)

    predictions_array = np.array([
        [prediction.get(label, 0.0) for label in labels]
        for prediction in list_label_and_scores
    ])
    predictions_array = predictions_array / predictions_array.sum(axis=1, keepdims=True)

    mean_prediction = predictions_array.mean(axis=0)

    entropy_mean, mean_entropy_individual, mutual_information = compute_entropy_metrics(predictions_array)

    variance_per_class, mean_variance = compute_variance(predictions_array)

    jsd_values, mean_jsd = compute_jsd(predictions_array)

    if debug:
        print(f"\nText: {text}")
        print(f"Labels: {labels}")
        print(f"Mean Prediction: {dict(zip(labels, mean_prediction))}")
        print(f"Entropy of Mean Prediction: {entropy_mean}")
        print(f"Mean Entropy of Individual Predictions: {mean_entropy_individual}")
        print(f"Mutual Information: {mutual_information}")
        print(f"Mean Variance: {mean_variance}")
        print(f"Mean JSD: {mean_jsd}")
        print(f"Combined Score: {combined_score}")
        print(f"Uncertainty: {uncertainty}")
        print(f"Variance (Tag): {variance_tag}, Variance (All): {variance_all}")
        print(f"Entropy (Tag): {entropy_tag}, Entropy (All): {entropy_all}")

    entropy_threshold = 0.4  
    variance_threshold = 0.05  
    jsd_threshold = 0.1  

    is_ambiguous = (
        entropy_mean > entropy_threshold or
        mean_variance > variance_threshold or
        mean_jsd > jsd_threshold
    )

    results = {
        'text': text,
        'mean_prediction': dict(zip(labels, mean_prediction)),
        'entropy_mean_prediction': entropy_mean,
        'mean_entropy_individual': mean_entropy_individual,
        'mutual_information': mutual_information,
        'mean_variance': mean_variance,
        'mean_jsd': mean_jsd,
        'combined_score': combined_score,
        'uncertainty': uncertainty,
        'variance_tag': variance_tag,
        'variance_all': variance_all,
        'entropy_tag': entropy_tag,
        'entropy_all': entropy_all,
        'is_ambiguous': is_ambiguous,
        'majority_vote_tag': majority_vote_tag
    }

    return results


def main():
    print("=== Text Ambiguity Detection Tool ===")

    # List available workloads (models) in saved_results directory or switch to models dir
    workloads = list_available_workloads()
    if not workloads:
        logger.error("No models found in 'saved_results' directory.")
        sys.exit(1)

    print("\nAvailable models:")
    for idx, workload_name in enumerate(workloads, start=1):
        print(f"{idx}. {workload_name}")

    while True:
        model_input = input("\nEnter the number of the model to use or type the model name: ").strip()
        if model_input.isdigit():
            model_idx = int(model_input)
            if 1 <= model_idx <= len(workloads):
                selected_model_name = workloads[model_idx - 1]
                break
            else:
                print("Invalid number. Please try again.")
        elif model_input in workloads:
            selected_model_name = model_input
            break
        else:
            print("Invalid input. Please try again.")

    verbose_input = input("Enable verbose output? Enter 'yes' or 'no' (default 'no'): ").strip().lower()
    if verbose_input in ['yes', 'y', 'true', '1']:
        verbose = True
    else:
        verbose = False

    eval_type_input = input("Enter evaluation type ('ensemble' or 'mc_dropout', default 'ensemble'): ").strip().lower()
    if eval_type_input in ['ensemble', 'mc_dropout']:
        eval_type = eval_type_input
    else:
        eval_type = 'ensemble'

    workload = Workload.load(workload_name=selected_model_name, load_models=True, quiet=True)
    logger.info(f"Loaded workload '{selected_model_name}' with {len(workload.models)} model(s).")

    print("\nYou can now enter texts to evaluate. Type 'exit' or press Enter on an empty line to quit.")

    while True:
        text = input("\nEnter the text to evaluate: ").strip()
        if not text:
            print("No text entered. Exiting.")
            break
        if text.lower() in ['exit', 'quit']:
            print("Exiting the tool.")
            break

        results = evaluate_text(workload, text, eval_type=eval_type, num_samples=100, debug=verbose)

        print("\n=== Evaluation Results ===")
        print(f"Text: {results['text']}")
        print(f"Predicted Probabilities: {results['mean_prediction']}")
        print(f"Predicted Label: {results['majority_vote_tag']}")
        print(f"Entropy of Mean Prediction: {results['entropy_mean_prediction']:.4f}")
        print(f"Mean Entropy of Individual Predictions: {results['mean_entropy_individual']:.4f}")
        print(f"Mutual Information: {results['mutual_information']:.4f}")
        print(f"Mean Variance: {results['mean_variance']:.4f}")
        print(f"Mean JSD: {results['mean_jsd']:.4f}")
        print(f"Ambiguous: {'Yes' if results['is_ambiguous'] else 'No'}")

    print("\nThank you for using the Text Ambiguity Detection Tool. Goodbye!")
if __name__ == "__main__":
    main()
