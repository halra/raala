import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix, accuracy_score, precision_score,
    recall_score, f1_score, roc_curve, auc
)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
import seaborn as sns
import logging
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from typing import List, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AmbiUtils:
    def threshold_based_ambiguity_detection(
        self,
        workload: pd.DataFrame,
        columns: list = ["variance_tag", "entropy_all", "jsd"],
        threshold_percentiles: dict = None,
        default_threshold_percentile: float = 90.0,
        debug: bool = False
    ) -> pd.DataFrame:
        thresholds = {}
        for column in columns:
            percentile = threshold_percentiles.get(column, default_threshold_percentile) if threshold_percentiles else default_threshold_percentile
            thresholds[column] = workload[column].quantile(percentile / 100.0)

        condition = pd.Series(False, index=workload.index)
        for column in columns:
            condition |= workload[column] > thresholds[column]

        workload['ambiguous'] = condition

        if debug:
            ambiguous_percentage = workload['ambiguous'].mean() * 100
            logger.debug(f"Ambiguous Samples: {ambiguous_percentage:.2f}% of the dataset")

        return workload

    def agreement_based_ambiguity_detection(
        self,
        df: pd.DataFrame,
        agreement_column: str = "highest_agreement",
        threshold_percentile: float = 60.0,
        debug: bool = False
    ) -> pd.DataFrame:
        if not (0 <= threshold_percentile <= 100):
            raise ValueError("threshold_percentile must be between 0 and 100.")

        threshold_value = df[agreement_column].quantile(threshold_percentile / 100.0)

        if debug:
            logger.debug(f"Threshold Percentile: {threshold_percentile}%")
            logger.debug(f"Threshold Value (computed from {agreement_column}): {threshold_value}")

        df['ambiguous_based_on_agreement'] = df[agreement_column] < threshold_value

        if debug:
            ambiguous_percentage = df['ambiguous_based_on_agreement'].mean() * 100
            logger.debug(f"Ambiguous Samples based on agreement: {ambiguous_percentage:.2f}% of the dataset")

        return df

    def test_detection(
        self,
        workload: pd.DataFrame,
        columns: list = ["variance_tag", "entropy_all", "jsd"],
        threshold_percentiles: dict = None,
        default_threshold_percentile: float = 90.0,
        agreement_column: str = "highest_agreement",
        threshold_percentile_agreement: float = 60.0,
        debug: bool = False,
        quiet: bool = False
    ) -> None:
        self.threshold_based_ambiguity_detection(
            workload=workload,
            columns=columns,
            threshold_percentiles=threshold_percentiles,
            default_threshold_percentile=default_threshold_percentile,
            debug=debug
        )
        self.agreement_based_ambiguity_detection(
            df=workload,
            agreement_column=agreement_column,
            threshold_percentile=threshold_percentile_agreement,
            debug=debug
        )

        df = workload
        true_positive = ((df['ambiguous_based_on_agreement']) & (df['ambiguous'])).sum()
        true_negative = ((~df['ambiguous_based_on_agreement']) & (~df['ambiguous'])).sum()
        total_matches = true_positive + true_negative
        total_samples = len(df)
        match_percentage = (total_matches / total_samples) * 100
        mismatch_percentage = 100 - match_percentage

        if not quiet:
            print("\nComparison Summary:")
            print("===================")
            print(f"Matching True (both True):   {true_positive} samples")
            print(f"Matching False (both False): {true_negative} samples")
            print(f"Total Matches:               {total_matches} samples ({match_percentage:.2f}%)")
            print(f"Mismatches:                  {total_samples - total_matches} samples ({mismatch_percentage:.2f}%)\n")

            print("True/False Counts per Column:")
            print("-----------------------------")
            print(f"True in 'ambiguous_based_on_agreement': {df['ambiguous_based_on_agreement'].sum()}")
            print(f"False in 'ambiguous_based_on_agreement': {(~df['ambiguous_based_on_agreement']).sum()}")
            print(f"True in 'ambiguous': {df['ambiguous'].sum()}")
            print(f"False in 'ambiguous': {(~df['ambiguous']).sum()}\n")

    def find_best_threshold_combination(
        self,
        workload: pd.DataFrame,
        columns: list = ["variance_tag", "entropy_all", "jsd"],
        min_threshold: int = 60,
        max_threshold: int = 99,
        min_agreement_threshold: int = 60,
        max_agreement_threshold: int = 99,
        agreement_column: str = 'highest_agreement',
        debug: bool = False
    ) -> tuple:
        best_combination = None
        highest_match_percentage = 0
        results = []

        for threshold in range(min_threshold, max_threshold + 1):
            for agreement_threshold in range(min_agreement_threshold, max_agreement_threshold + 1):
                df_copy = workload.copy()

                self.threshold_based_ambiguity_detection(
                    workload=df_copy,
                    columns=columns,
                    default_threshold_percentile=threshold,
                    debug=False
                )

                self.agreement_based_ambiguity_detection(
                    df=df_copy,
                    agreement_column=agreement_column,
                    threshold_percentile=agreement_threshold,
                    debug=False
                )

                matches = (df_copy['ambiguous'] == df_copy['ambiguous_based_on_agreement']).sum()
                match_percentage = (matches / len(df_copy)) * 100

                if match_percentage > highest_match_percentage:
                    highest_match_percentage = match_percentage
                    best_combination = (threshold, agreement_threshold)

                results.append({
                    'Threshold': threshold,
                    'Agreement Threshold': agreement_threshold,
                    'Match %': match_percentage
                })

                if debug:
                    logger.debug(f"Threshold: {threshold}, Agreement Threshold: {agreement_threshold}, Match %: {match_percentage:.2f}%")

        results_df = pd.DataFrame(results)

        print(f"\nBest Threshold Combination: Threshold = {best_combination[0]}, Agreement Threshold = {best_combination[1]}")
        print(f"Highest Match Percentage: {highest_match_percentage:.2f}%")
        return best_combination, highest_match_percentage, results_df

    def show_confusion_matrix(
        self,
        workload: pd.DataFrame,
        y_true_col: str = 'ambiguous_based_on_agreement',
        y_pred_col: str = 'ambiguous',
        plot: bool = True
    ) -> None:
        y_true = workload[y_true_col]
        y_pred = workload[y_pred_col]
        cm = confusion_matrix(y_true, y_pred)

        print("Confusion Matrix:")
        print(cm)
        if plot:
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.title('Confusion Matrix')
            plt.ylabel('Actual Labels')
            plt.xlabel('Predicted Labels')
            plt.show()

            cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues')
            plt.title('Normalized Confusion Matrix')
            plt.ylabel('Actual Labels')
            plt.xlabel('Predicted Labels')
            plt.show()

    def show_eval_metrics(
        self,
        workload: pd.DataFrame,
        y_true_col: str = 'ambiguous_based_on_agreement',
        y_pred_col: str = 'ambiguous'
    ) -> tuple:
        y_true = workload[y_true_col]
        y_pred = workload[y_pred_col]
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)

        print(f"Accuracy: {accuracy:.2f}")
        print(f"Precision: {precision:.2f}")
        print(f"Recall: {recall:.2f}")
        print(f"F1 Score: {f1:.2f}")
        return accuracy, precision, recall, f1

    def show_roc_auc_curve(
        self,
        workload: pd.DataFrame,
        y_true_col: str = 'ambiguous_based_on_agreement',
        predictors: list = ['entropy_all', 'jsd', 'variance_tag', 'score'],
        plot: bool = True
    ) -> None:
        y_true = workload[y_true_col]

        if plot:
            plt.figure(figsize=(10, 8))

        for predictor in predictors:
            y_scores = workload[predictor]
            fpr, tpr, _ = roc_curve(y_true, y_scores)
            roc_auc = auc(fpr, tpr)
            if plot:
                plt.plot(fpr, tpr, label=f'{predictor} (AUC = {roc_auc:.2f})')
            logger.info(f'{predictor} (AUC = {roc_auc:.2f})')

        if plot:
            plt.plot([0, 1], [0, 1], 'k--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('ROC Curves for Ambiguity Detection')
            plt.legend(loc='lower right')
            plt.show()

    def show_combined_auc(
        self,
        workload: pd.DataFrame,
        y_col: str = 'ambiguous_based_on_agreement',
        predictors: list = ['entropy_all', 'jsd', 'variance_tag', 'score']
    ) -> None:
        X = workload[predictors]
        y = workload[y_col]
        model = LogisticRegression(max_iter=1000)
        scores = cross_val_score(model, X, y, cv=5, scoring='roc_auc')
        mean_auc = np.mean(scores)
        logger.info(f'Combined model AUC: {mean_auc:.2f}')
        print(f'Combined model AUC: {mean_auc:.2f}')

    def calculate_optimal_predictor_threshold(
        self,
        workload: pd.DataFrame,
        y_col: str = 'ambiguous_based_on_agreement',
        predictors: list = ['entropy_all', 'jsd', 'variance_tag', 'score']
    ) -> None:
        y_true = workload[y_col]
        for predictor in predictors:
            y_scores = workload[predictor]
            fpr, tpr, thresholds = roc_curve(y_true, y_scores)
            youdens_j = tpr - fpr
            optimal_idx = np.argmax(youdens_j)
            optimal_threshold = thresholds[optimal_idx]
            print(f'Optimal threshold for {predictor}: {optimal_threshold:.4f}')

    def define_ambiguity(
        self,
        df: pd.DataFrame,
        agreement_threshold: float = 70.0,
        entropy_threshold: Optional[float] = None
    ) -> pd.DataFrame:
        if entropy_threshold is not None:
            df['ambiguous'] = (
                (df['highest_agreement'] < agreement_threshold) |
                (df['entropy'] > entropy_threshold)
            )
        else:
            df['ambiguous'] = df['highest_agreement'] < agreement_threshold

        return df

    def normalize_uncertainty_measures(
        self,
        df: pd.DataFrame,
        uncertainty_columns: list,
        method: str = 'minmax'
    ) -> pd.DataFrame:
        if method == 'minmax':
            scaler = MinMaxScaler()
        elif method == 'standard':
            scaler = StandardScaler()
        else:
            raise ValueError("Unsupported normalization method. Choose 'minmax' or 'standard'.")

        df_normalized = df.copy()
        df_normalized[uncertainty_columns] = scaler.fit_transform(df[uncertainty_columns])

        return df_normalized
    def find_best_threshold_combination_based_on_error_rate(
        self,
        workload: pd.DataFrame,
        columns: list = ["variance_tag", "entropy_all", "jsd"],
        min_threshold: int = 60,
        max_threshold: int = 99,
        min_agreement_threshold: int = 70,
        max_agreement_threshold: int = 70,
        agreement_column: str = 'highest_agreement',
        debug: bool = False
    ) -> tuple:
 
        best_threshold_combination = None
        lowest_error_rate = 100  # Start with the maximum error rate (100%)
        results = []
        for threshold in range(min_threshold, max_threshold + 1):
            for agreement_threshold in range(min_agreement_threshold, max_agreement_threshold + 1):
                df_copy = workload.copy()

                self.threshold_based_ambiguity_detection(
                    workload=df_copy,
                    columns=columns,
                    default_threshold_percentile=threshold,
                    debug=False
                )

                self.agreement_based_ambiguity_detection(
                    df=df_copy,
                    agreement_column=agreement_column,
                    threshold_percentile=agreement_threshold,
                    debug=False
                )

                filtered_df = df_copy[
                    (df_copy['ambiguous'] == True) & (df_copy['ambiguous_based_on_agreement'] == True)
                ]
                filtered_df_neg = df_copy[
                    (df_copy['ambiguous'] == False) & (df_copy['ambiguous_based_on_agreement'] == False)
                ]
                matched_rows = filtered_df.shape[0] + filtered_df_neg.shape[0]
                percentage = (matched_rows / len(df_copy)) * 100

                error_rate = 100 - percentage
                if error_rate < lowest_error_rate:
                    lowest_error_rate = error_rate
                    best_threshold_combination = (threshold, agreement_threshold)

                results.append({
                    'Threshold': threshold,
                    'Agreement Threshold': agreement_threshold,
                    'Matched Rows %': percentage,
                    'Error Rate %': error_rate
                })

                #Debug
                if debug:
                    logger.debug(f"Threshold: {threshold}, Agreement Threshold: {agreement_threshold}")
                    logger.debug(f"Matched Rows Percentage: {percentage:.2f}%")
                    logger.debug(f"Error Rate: {error_rate:.2f}%")

        results_df = pd.DataFrame(results)

        if debug:
            print("\nSummary of All Threshold Combinations:")
            print(results_df)

        if best_threshold_combination:
            print(f"\nBest Threshold Combination: Threshold = {best_threshold_combination[0]}, "
                f"Agreement Threshold = {best_threshold_combination[1]}")
            print(f"Lowest Error Rate: {lowest_error_rate:.2f}%")
        else:
            print("\nNo valid threshold combination found.")

        return best_threshold_combination, lowest_error_rate, results_df
