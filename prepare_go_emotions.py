import os
import logging
from datetime import datetime
from typing import List, Optional

import pandas as pd
import numpy as np
from scipy.stats import entropy
from scipy.spatial.distance import jensenshannon

from domain.workload import Workload
from helper.workload_utils import WorkloadUtils

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

console_handler = logging.StreamHandler()
file_handler = logging.FileHandler('./Logs/goemotions_processing.log')
console_handler.setLevel(logging.INFO)
file_handler.setLevel(logging.DEBUG)  

formatter = logging.Formatter('%(asctime)s %(levelname)s:%(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)


def file_exists(file_path: str) -> bool:

    exists = os.path.exists(file_path)
    logger.debug(f"Checking if file exists at {file_path}: {exists}")
    return exists


def dataset_stats_finalized(df: pd.DataFrame, columns: List[str]) -> None:

    logger.info("Calculating finalized dataset statistics...")
    if all(col in df.columns for col in columns):
        summed_series = df[columns].sum(axis=1)
        average_sum = summed_series.mean()
        logger.info(f"Average annotations per sample: {average_sum:.2f}")
    else:
        missing_cols = [col for col in columns if col not in df.columns]
        logger.warning(f"Columns not found for finalized stats: {missing_cols}")


def validate_and_load_parts(current_dir: str) -> List[pd.DataFrame]:

    go_emotions_parts = []
    for i in range(3):
        emotions_path = os.path.join(
            current_dir,
            "datasets",
            "go_emotions",
            f"goemotions_{i + 1}.csv"
        )
        if not file_exists(emotions_path):
            logger.error(
                f"File not found: {emotions_path}. Please ensure you have downloaded the datasets "
                "'goemotions_1.csv', 'goemotions_2.csv', 'goemotions_3.csv' "
                "from https://github.com/google-research/google-research/tree/master/goemotions/data/full_dataset "
                "and placed them in the datasets/go_emotions folder."
            )
            raise FileNotFoundError(f"Dataset file not found at {emotions_path}")
        try:
            df_part = pd.read_csv(emotions_path)
            go_emotions_parts.append(df_part)
            logger.info(f"Loaded dataset part: {emotions_path} with shape {df_part.shape}")
            logger.debug(f"Columns in {emotions_path}: {df_part.columns.tolist()}")
        except Exception as e:
            logger.exception(f"An error occurred while reading {emotions_path}: {e}")
            raise
    return go_emotions_parts


def merge_and_cleanup_data(go_emotions_parts: List[pd.DataFrame]) -> pd.DataFrame:

    concatenated_df = pd.concat(go_emotions_parts, ignore_index=True)
    logger.info(f"Dataset parts merged successfully. Merged shape: {concatenated_df.shape}")

    if 'is_error' in concatenated_df.columns:
        initial_row_count = concatenated_df.shape[0]
        concatenated_df = concatenated_df[~concatenated_df['is_error']].copy()
        rows_removed = initial_row_count - concatenated_df.shape[0]
        logger.info(f"Removed {rows_removed} error rows from the dataset. Remaining rows: {concatenated_df.shape[0]}")
    else:
        logger.warning("'is_error' column not found in the DataFrame.")

    return concatenated_df


def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:

    logger.info("Starting DataFrame preprocessing...")
    initial_shape = df.shape
    logger.debug(f"Initial DataFrame shape: {initial_shape}")

    emotion_labels = [
        'admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring',
        'confusion', 'curiosity', 'desire', 'disappointment', 'disapproval',
        'disgust', 'embarrassment', 'excitement', 'fear', 'gratitude', 'grief',
        'joy', 'love', 'nervousness', 'optimism', 'pride', 'realization',
        'relief', 'remorse', 'sadness', 'surprise', 'neutral'
    ]

    missing_labels = [label for label in emotion_labels if label not in df.columns]
    if missing_labels:
        logger.error(f"Missing emotion labels in the dataset: {missing_labels}")
        raise ValueError(f"Dataset is missing columns: {missing_labels}")

    logger.info("Creating 'label' column with assigned emotions.")
    df['label'] = df[emotion_labels].apply(
        lambda row: ','.join([label for label in emotion_labels if row[label] == 1]),
        axis=1
    )

    initial_row_count = df.shape[0]
    df = df[~df['label'].str.contains(',')].copy()
    df = df[df['label'].astype(str).str.strip() != ''].copy()
    rows_removed = initial_row_count - df.shape[0]
    logger.info(f"Filtered to single-label data. Removed {rows_removed} rows with multiple labels.")

    columns_to_drop = [col for col in df.columns if col in emotion_labels or col == 'is_error']
    df = df.drop(columns=columns_to_drop, errors='ignore').copy()
    logger.debug(f"Dropped columns: {columns_to_drop}")

    df.reset_index(drop=True, inplace=True)

    logger.info(f"DataFrame preprocessing completed. Final shape: {df.shape}")
    return df


def create_workload_struct(df: pd.DataFrame, source_path: str) -> Workload:

    labels = df['label'].unique().tolist()
    workload_struct = Workload(
        dataset_path="df",
        created_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        labels=labels,
        name="go_emotions",
        source_path=source_path
    )
    logger.info(f"Workload structure created with {len(labels)} labels: {labels}")
    return workload_struct


def add_agreement_metrics(df: pd.DataFrame, labels: List[str]) -> pd.DataFrame:

    logger.info("Adding agreement metrics to the DataFrame...")
    prob_labels = [f"{label}_probability" for label in labels]

    missing_columns = [col for col in prob_labels if col not in df.columns]
    if missing_columns:
        logger.debug("Probability columns not found. Creating probability columns.")
        total_counts = df[labels].sum(axis=1)
        for label in labels:
            if label in df.columns:
                df[f"{label}_probability"] = df[label] / total_counts.replace(0, np.nan)
            else:
                df[f"{label}_probability"] = 0.0
                logger.warning(f"Label column '{label}' not found. Setting probability to 0.")
        df[prob_labels] = df[prob_labels].fillna(0)
    else:
        logger.debug("Probability columns found. Proceeding with normalization.")

    df[prob_labels] = df[prob_labels].div(df[prob_labels].sum(axis=1), axis=0)
    logger.debug("Normalized label probabilities.")

    df['highest_agreement'] = df[prob_labels].max(axis=1)
    logger.debug("Calculated highest agreement.")

    df['entropy_agreement'] = df[prob_labels].apply(lambda x: entropy(x), axis=1)
    logger.debug("Calculated entropy agreement.")

    df['variance_agreement'] = df[prob_labels].var(axis=1)
    logger.debug("Calculated variance agreement.")

    uniform_distribution = np.full(len(prob_labels), 1 / len(prob_labels))
    df['jsd_agreement'] = df[prob_labels].apply(
        lambda x: jensenshannon(x, uniform_distribution, base=2), axis=1
    )
    logger.debug("Calculated Jensen-Shannon divergence.")

    logger.info("Agreement metrics added successfully.")
    return df


def save_workload_struct(work_dir: str, workload_struct: Workload) -> None:

    struct_data_path = os.path.join(work_dir, 'struct_data.json')
    try:
        with open(struct_data_path, 'w') as f:
            f.write(workload_struct.to_json())
        logger.info(f"Workload structure saved to {struct_data_path}")
    except Exception as e:
        logger.exception(f"Failed to save workload structure: {e}")
        raise


def dataset_stats(df: pd.DataFrame, worker_id: Optional[str], label_columns: List[str]) -> None:

    logger.info("Calculating dataset statistics...")
    if worker_id and worker_id in df.columns:
        num_annotators = df[worker_id].nunique()
        logger.info(f"Number of annotators: {num_annotators}")
    else:
        logger.warning("Worker ID column not found or not provided.")

    if label_columns:
        if all(label in df.columns for label in label_columns):
            df['total_votes'] = df[label_columns].sum(axis=1)
            average_votes_per_sample = df['total_votes'].mean()
            logger.info(f"Average votes per sample: {average_votes_per_sample:.2f}")

            votes_per_text = df.groupby('text')[label_columns].sum()
            votes_per_text['total_votes'] = votes_per_text.sum(axis=1)
            average_votes_per_text = votes_per_text['total_votes'].mean()
            logger.info(f"Average total votes per text: {average_votes_per_text:.2f}")
        else:
            missing_labels = [label for label in label_columns if label not in df.columns]
            logger.warning(f"Label columns not found in DataFrame: {missing_labels}")
    else:
        logger.warning("Label columns not provided or empty.")


def prepare_go_emotions_dataset() -> None:

    logger.info("Starting GoEmotions dataset preparation...")

    current_dir = os.getcwd()
    go_emotions_parts = validate_and_load_parts(current_dir)

    work_dir = os.path.join(current_dir, "workload")
    os.makedirs(work_dir, exist_ok=True)
    logger.info(f"Work directory created at {work_dir}")

    df = merge_and_cleanup_data(go_emotions_parts)
    dataset_stats(df.copy(), worker_id=None, label_columns=[
        'admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring',
        'confusion', 'curiosity', 'desire', 'disappointment', 'disapproval',
        'disgust', 'embarrassment', 'excitement', 'fear', 'gratitude', 'grief',
        'joy', 'love', 'nervousness', 'optimism', 'pride', 'realization',
        'relief', 'remorse', 'sadness', 'surprise', 'neutral'
    ])
    df = preprocess_dataframe(df)

    source_path = os.path.join(
        current_dir,
        "datasets",
        "go_emotions",
        "goemotions_1.csv"
    )
    workload_struct = create_workload_struct(df, source_path)

    w = WorkloadUtils(workload_struct, df, work_dir)
    logger.info("Initialized WorkloadUtils.")
    w.workload_prepare()
    logger.info("Workload preparation completed.")

    w.df = add_agreement_metrics(w.df, workload_struct.labels)
    w.generate_train_test_dev_set()
    logger.info("Train, test, and dev sets generated.")

    final_df_path = os.path.join(work_dir, 'df.csv')
    w.df.to_csv(final_df_path, index=False)
    logger.info(f"Final DataFrame saved to {final_df_path}")
    logger.info(f"Final DataFrame shape: {w.df.shape}")

    save_workload_struct(work_dir, workload_struct)
    dataset_stats(w.df.copy(), worker_id=None, label_columns=workload_struct.labels)
    logger.info("GoEmotions dataset preparation completed successfully.")


if __name__ == "__main__":
    try:
        prepare_go_emotions_dataset()
    except Exception as e:
        logger.exception(f"An error occurred during dataset preparation: {e}")
