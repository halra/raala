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
file_handler = logging.FileHandler('./Logs/rotten_tomatoes_processing.log')
console_handler.setLevel(logging.INFO)
file_handler.setLevel(logging.INFO)

formatter = logging.Formatter('%(asctime)s %(levelname)s:%(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)


def file_exists(file_path: str) -> bool:

    exists = os.path.exists(file_path)
    logger.debug(f"Checking if file exists at {file_path}: {exists}")
    return exists


def dataset_stats(
    df: pd.DataFrame,
    worker_id: Optional[str],
    label_columns: Optional[List[str]]
) -> None:

    logger.info("Calculating dataset statistics...")
    if worker_id and worker_id in df.columns:
        num_annotators = df[worker_id].nunique()
        logger.info(f"Number of annotators: {num_annotators}")

        num_samples = df['Input.original_sentence'].nunique()
        logger.info(f"Number of unique samples: {num_samples}")

        annotations_per_sample = df.groupby('Input.original_sentence')[worker_id].nunique().mean()
        logger.info(f"Average number of annotations per sample: {annotations_per_sample:.2f}")
    else:
        logger.warning(f"Worker ID column '{worker_id}' not found in DataFrame.")

    if label_columns:
        if all(label in df.columns for label in label_columns):
            annotations_per_label = df[label_columns].sum()
            logger.info("Annotations per label:")
            for label, count in annotations_per_label.items():
                logger.debug(f"  {label}: {count}")
        else:
            missing_labels = [label for label in label_columns if label not in df.columns]
            logger.warning(f"Label columns not found in DataFrame: {missing_labels}")
    else:
        logger.warning("No label columns provided for dataset statistics.")


def dataset_stats_finalized(df: pd.DataFrame, columns: List[str]) -> None:

    logger.info("Calculating finalized dataset statistics...")
    if all(col in df.columns for col in columns):
        summed_series = df[columns].sum(axis=1)
        average_sum = summed_series.mean()
        logger.info(f"Average annotations per sample: {average_sum:.2f}")
    else:
        missing_cols = [col for col in columns if col not in df.columns]
        logger.warning(f"Columns not found for finalized stats: {missing_cols}")


def prepare_rotten_tomatoes_dataset() -> None:

    logger.info("Starting Rotten Tomatoes dataset preparation...")
    current_dir = os.getcwd()
    rt_path = os.path.join(current_dir, "datasets", "rotten_tomatoes", "RT_base.csv")

    if not file_exists(rt_path):
        logger.error(
            f"File not found: {rt_path}. Please ensure you have downloaded the dataset "
            "'RT_base.csv' from https://huggingface.co/datasets/rotten_tomatoes or the reduced dataset from  http://fprodrigues.com//mturk-datasets.tar.gz"
            "and placed it in the datasets/rotten_tomatoes folder."
        )
        raise FileNotFoundError(f"Dataset not found at {rt_path}")

    work_dir = os.path.join(current_dir, "workload")
    os.makedirs(work_dir, exist_ok=True)
    logger.info(f"Work directory created at {work_dir}")

    try:
        df = pd.read_csv(rt_path)
        logger.info(f"Dataset loaded from {rt_path}")
        logger.info(f"Dataset shape: {df.shape}")
        logger.debug(f"Dataset columns: {df.columns.tolist()}")
    except Exception as e:
        logger.exception(f"An error occurred while reading the dataset: {e}")
        raise

    dataset_stats(df, worker_id="WorkerId", label_columns=['Answer.sent'])

    df = preprocess_dataframe(df)

    workload_struct = create_workload_struct(df, rt_path)

    w = WorkloadUtils(workload_struct, df, work_dir)
    logger.info("Initialized WorkloadUtils.")
    w.workload_prepare(max_label_count=100000)
    logger.info("Workload preparation completed.")

    w.df = add_agreement_metrics(w.df, workload_struct.labels)

    final_df_path = os.path.join(work_dir, 'df.csv')
    w.df.to_csv(final_df_path, index=False)
    logger.info(f"Final DataFrame saved to {final_df_path}")
    logger.info(f"Final DataFrame shape: {w.df.shape}")

    w.generate_train_test_dev_set()
    logger.info("Train, test, and dev sets generated.")

    save_workload_struct(work_dir, workload_struct)

    dataset_stats(w.df, worker_id=None, label_columns=workload_struct.labels)
    dataset_stats_finalized(w.df, workload_struct.labels)
    logger.info("Rotten Tomatoes dataset preparation completed successfully.")


def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:

    logger.info("Starting DataFrame preprocessing...")
    original_columns = df.columns.tolist()
    logger.debug(f"Original columns: {original_columns}")

    df = df.rename(columns={
        "Input.true_sent": 'gold_label',
        "Input.id": 'id',
        "Input.original_sentence": 'text',
        "Answer.sent": 'label'
    }, errors='ignore')
    logger.debug(f"Columns after renaming: {df.columns.tolist()}")

    df = df.drop(columns=["Input.stemmed_sent", "WorkerId"], errors='ignore')
    logger.debug(f"Columns after dropping unnecessary ones: {df.columns.tolist()}")

    columns_order = ['text', 'id', 'label', 'gold_label']
    df = df[[col for col in columns_order if col in df.columns]]
    logger.info(f"Columns after reordering: {df.columns.tolist()}")
    logger.info(f"DataFrame shape after preprocessing: {df.shape}")

    return df


def create_workload_struct(df: pd.DataFrame, source_path: str) -> Workload:

    labels = df['label'].unique().tolist()
    workload_struct = Workload(
        dataset_path="df",
        created_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        labels=labels,
        name="rotten_tomatoes",
        source_path=source_path
    )
    logger.info(f"Workload structure created with labels: {labels}")
    return workload_struct


def add_agreement_metrics(df: pd.DataFrame, labels: List[str]) -> pd.DataFrame:

    logger.info("Adding agreement metrics to the DataFrame...")

    for label in labels:
        if label not in df.columns:
            df[label] = (df['label'] == label).astype(int)
            logger.debug(f"Created count column for label '{label}'")

    df['total_votes'] = df[labels].sum(axis=1)

    initial_row_count = df.shape[0]
    df = df[df['total_votes'] >= 2].copy()
    num_rows_removed = initial_row_count - df.shape[0]
    logger.info(f"Removed {num_rows_removed} rows with less than 2 votes. Remaining rows: {df.shape[0]}")

    for label in labels:
        df[f"{label}_probability"] = df[label] / df['total_votes']
        logger.debug(f"Calculated probability for label '{label}'")

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


def save_workload_struct(work_dir: str, workload_struct: Workload) -> None:

    struct_data_path = os.path.join(work_dir, 'struct_data.json')
    try:
        with open(struct_data_path, 'w') as f:
            f.write(workload_struct.to_json())
        logger.info(f"Workload structure saved to {struct_data_path}")
    except Exception as e:
        logger.exception(f"Failed to save workload structure: {e}")
        raise


if __name__ == "__main__":
    try:
        prepare_rotten_tomatoes_dataset()
    except Exception as e:
        logger.exception(f"An error occurred during dataset preparation: {e}")
