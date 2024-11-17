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
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
file_handler = logging.FileHandler('./Logs/gab_hate_corpus_processing.log')
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


def dataset_stats(df: pd.DataFrame, worker_id: Optional[str], label_columns: Optional[List[str]]) -> None:

    logger.info("Calculating initial dataset statistics...")
    if worker_id and worker_id in df.columns:
        num_annotators = df[worker_id].nunique()
        num_samples = df.shape[0]
        logger.info(f"Number of annotators: {num_annotators}")
        logger.info(f"Number of samples: {num_samples}")
    else:
        logger.warning(f"Worker ID column '{worker_id}' not found in DataFrame.")

    if label_columns:
        missing_labels = [label for label in label_columns if label not in df.columns]
        if missing_labels:
            logger.warning(f"Label columns not found in DataFrame: {missing_labels}")
        else:
            label_counts = df[label_columns].sum()
            logger.info("Label counts:")
            for label, count in label_counts.items():
                logger.info(f"  {label}: {count}")
    else:
        logger.warning("No label columns provided for dataset statistics.")


def dataset_stats_finalized(df: pd.DataFrame, columns: List[str]) -> None:

    logger.info("Calculating finalized dataset statistics...")
    if all(col in df.columns for col in columns):
        summed_series = df[columns].sum(axis=1)
        average_annotations = summed_series.mean()
        logger.info(f"Average annotations per sample: {average_annotations:.2f}")
    else:
        missing_cols = [col for col in columns if col not in df.columns]
        logger.warning(f"Columns not found in DataFrame for finalized stats: {missing_cols}")


def get_dataset_path() -> str:

    current_dir = os.getcwd()
    dataset_path = os.path.join(
        current_dir,
        "datasets",
        "gab_hate_corpus",
        "GabHateCorpus_annotations.tsv"
    )
    logger.debug(f"Dataset path: {dataset_path}")
    return dataset_path


def create_work_directory() -> str:

    work_dir = os.path.join(os.getcwd(), "workload")
    os.makedirs(work_dir, exist_ok=True)
    logger.info(f"Work directory created at {work_dir}")
    return work_dir


def load_dataset(dataset_path: str) -> pd.DataFrame:

    logger.info(f"Loading dataset from {dataset_path}")
    if not file_exists(dataset_path):
        logger.error(
            f"File not found: {dataset_path}. Please ensure you have downloaded the dataset "
            "'GabHateCorpus_annotations.tsv' from https://osf.io/edua3/ "
            "and placed it in the datasets/gab_hate_corpus folder."
        )
        raise FileNotFoundError(f"Dataset not found at {dataset_path}")
    try:
        df = pd.read_csv(dataset_path, sep='\t')
        logger.info(f"Dataset loaded successfully with shape {df.shape}")
        logger.debug(f"Dataset columns: {df.columns.tolist()}")
    except Exception as e:
        logger.exception(f"An error occurred while reading the dataset: {e}")
        raise
    return df


def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:

    logger.info("Starting DataFrame preprocessing...")
    logger.debug(f"Initial DataFrame shape: {df.shape}")
    logger.debug(f"Initial DataFrame columns: {df.columns.tolist()}")

    df = df.rename(columns={
        "ID": 'id',
        "Text": 'text',
        "Hate": 'label'
    }, errors='ignore')
    logger.debug(f"Columns after renaming: {df.columns.tolist()}")

    columns_to_drop = [
        'HD', 'CV', 'VO', 'REL', 'RAE', 'SXO', 'GEN',
        'IDL', 'NAT', 'POL', 'MPH', 'EX', 'IM'
    ]
    df = df.drop(columns=columns_to_drop, errors='ignore')
    logger.debug(f"Columns after dropping unnecessary ones: {df.columns.tolist()}")

    if 'label' in df.columns:
        df['label'] = df['label'].map({0: 'no_hate', 1: 'hate'})
        logger.info("Label values mapped to 'no_hate' and 'hate'.")

    columns_order = [col for col in ['text', 'id', 'label'] if col in df.columns]
    df = df[columns_order]
    logger.debug(f"Columns after reordering: {df.columns.tolist()}")

    logger.info(f"DataFrame preprocessing completed. Final shape: {df.shape}")
    return df


def create_workload_struct(df: pd.DataFrame, source_path: str) -> Workload:

    labels = df['label'].unique().tolist()
    workload_struct = Workload(
        dataset_path="df",
        created_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        labels=labels,
        name="gab_hate_corpus",
        source_path=source_path
    )
    logger.info(f"Workload structure created with labels: {labels}")
    return workload_struct


def process_workload(workload_struct: Workload, df: pd.DataFrame, work_dir: str) -> WorkloadUtils:

    logger.info("Initializing WorkloadUtils...")
    w = WorkloadUtils(workload_struct, df, work_dir)
    w.workload_prepare(max_label_count=100000)
    logger.info("Workload preparation completed.")
    return w


def balance_classes(df: pd.DataFrame, label_column: str) -> pd.DataFrame:

    logger.info(f"Balancing classes based on '{label_column}' column...")
    class_counts = df[label_column].value_counts()
    logger.info(f"Original class counts:\n{class_counts}")

    min_count = class_counts.min()
    logger.info(f"Minimum class count for balancing: {min_count}")

    balanced_df = df.groupby(label_column, group_keys=False).apply(
        lambda x: x.sample(min_count, random_state=42)
    ).reset_index(drop=True)

    balanced_class_counts = balanced_df[label_column].value_counts()
    logger.info(f"Balanced class counts:\n{balanced_class_counts}")

    logger.info(f"DataFrame shape after balancing: {balanced_df.shape}")
    return balanced_df


def display_class_counts(df: pd.DataFrame, label_column: str) -> None:

    logger.info(f"Displaying class counts for '{label_column}' column...")
    class_counts = df[label_column].value_counts()
    for label, count in class_counts.items():
        logger.info(f"  {label}: {count}")
    min_count = class_counts.min()
    logger.info(f"Minimum count among '{label_column}' classes: {min_count}")


def add_agreement_metrics(df: pd.DataFrame, labels: List[str]) -> pd.DataFrame:

    logger.info("Adding agreement metrics to the DataFrame...")
    missing_labels = [label for label in labels if label not in df.columns]
    if missing_labels:
        logger.error(f"Label columns not found in DataFrame: {missing_labels}")
        raise ValueError(f"Missing label columns: {missing_labels}")

    df['total_votes'] = df[labels].sum(axis=1)
    logger.debug("Calculated total votes per sample.")

    initial_row_count = df.shape[0]
    df = df[df['total_votes'] >= 2].copy()
    rows_removed = initial_row_count - df.shape[0]
    logger.info(f"Removed {rows_removed} rows with less than 2 votes. Remaining rows: {df.shape[0]}")

    for label in labels:
        df[f"{label}_probability"] = df[label] / df['total_votes']
        logger.debug(f"Calculated probability for label '{label}'.")

    prob_labels = [f"{label}_probability" for label in labels]

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


def save_workload_data(w: WorkloadUtils, work_dir: str) -> None:

    logger.info("Saving workload data and structure...")
    try:
        df_path = os.path.join(work_dir, 'df.csv')
        w.df.to_csv(df_path, index=False)
        logger.info(f"Workload DataFrame saved to {df_path}")
        w.generate_train_test_dev_set()
        logger.info("Train, test, and dev sets generated.")
        save_workload_struct(work_dir, w.workload)
    except Exception as e:
        logger.exception(f"An error occurred while saving workload data: {e}")
        raise


def save_workload_struct(work_dir: str, workload_struct: Workload) -> None:

    struct_data_path = os.path.join(work_dir, 'struct_data.json')
    try:
        with open(struct_data_path, 'w') as f:
            f.write(workload_struct.to_json())
        logger.info(f"Workload structure saved to {struct_data_path}")
    except Exception as e:
        logger.exception(f"Failed to save workload structure: {e}")
        raise


def prepare_gab_hate_corpus_dataset() -> None:

    logger.info("Starting GabHateCorpus dataset preparation...")
    dataset_path = get_dataset_path()
    work_dir = create_work_directory()
    df = load_dataset(dataset_path)
    dataset_stats(df, worker_id="Annotator", label_columns=["Hate"])
    df = preprocess_dataframe(df)
    workload_struct = create_workload_struct(df, dataset_path)
    w = process_workload(workload_struct, df, work_dir)
    w.df = balance_classes(w.df, 'gold_label')
    display_class_counts(w.df, 'gold_label')
    w.df = add_agreement_metrics(w.df, workload_struct.labels)
    save_workload_data(w, work_dir)
    dataset_stats_finalized(w.df, workload_struct.labels)
    logger.info("GabHateCorpus dataset preparation completed successfully.")


if __name__ == "__main__":
    try:
        prepare_gab_hate_corpus_dataset()
    except Exception as e:
        logger.exception(f"An error occurred during dataset preparation: {e}")
