import os
import logging
from typing import Any, Optional, List

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class WorkloadUtils:
    def __init__(self, workload: Any, df: pd.DataFrame, workdir: str):
        self.workload = workload
        self.df = df
        self.workdir = workdir
        self.sub_dfs_train: Optional[pd.DataFrame] = None
        self.sub_dfs_test: Optional[pd.DataFrame] = None
        self.sub_dfs_dev: Optional[pd.DataFrame] = None

    def workload_prepare(self, max_label_count: int = 3000) -> pd.DataFrame:
        logger.info("Counting (id, label) occurrences")
        self._count_and_pivot()
        self._remove_single_votes()
        self._add_label_probabilities()
        self._add_gold_label()
        self._balance_label_distribution(max_label_count)
        return self.df

    def _count_and_pivot(self) -> None:
        if 'id' not in self.df.columns or 'label' not in self.df.columns:
            raise ValueError("DataFrame must contain 'id' and 'label' columns.")

        grouped_df = self.df.groupby(['id', 'label']).size().reset_index(name='count')
        pivot_df = grouped_df.pivot(index='id', columns='label', values='count').fillna(0).reset_index()

        self.df = pd.merge(
            self.df.drop_duplicates(subset=['id']), pivot_df, on='id', how='left'
        )

    def _remove_single_votes(self) -> None:
        vote_counts = self.df[self.workload.labels].sum(axis=1)
        single_vote_rows = vote_counts < 2
        num_single_vote_rows = single_vote_rows.sum()
        initial_row_count = self.df.shape[0]
        self.df = self.df.loc[~single_vote_rows]
        logger.info(f"Removed {num_single_vote_rows} rows with single votes out of {initial_row_count}")
        logger.info(f"Remaining rows after removal: {self.df.shape[0]}")

    def _add_label_probabilities(self) -> None:
        label_columns = self.workload.labels
        missing_labels = [label for label in label_columns if label not in self.df.columns]
        if missing_labels:
            raise ValueError(f"Label columns missing from DataFrame: {missing_labels}")

        total_counts = self.df[label_columns].sum(axis=1).replace(0, np.nan)
        probabilities = self.df[label_columns].div(total_counts, axis=0).fillna(0)
        probability_columns = [f'{label}_probability' for label in label_columns]
        probabilities.columns = probability_columns

        self.df = pd.concat([self.df, probabilities], axis=1)
        logger.info(f"Added label probabilities for labels: {', '.join(label_columns)}")

    def _add_gold_label(self) -> None:
        if 'gold_label' not in self.df.columns:
            logger.info("Adding gold label based on highest empirical probability")
            probability_columns = [f'{label}_probability' for label in self.workload.labels]
            self.df['gold_label'] = self.df[probability_columns].idxmax(axis=1).str.replace('_probability', '')

            if 'label' in self.df.columns:
                label_idx = self.df.columns.get_loc('label') + 1
            else:
                label_idx = 3
            cols = list(self.df.columns)
            cols.insert(label_idx, cols.pop(cols.index('gold_label')))
            self.df = self.df[cols]

    def _balance_label_distribution(self, max_label_count: int = 3000) -> None:
        logger.info("Balancing label distribution")
        self.df = self.df.groupby('gold_label').apply(
            lambda x: x.sample(n=min(len(x), max_label_count), random_state=42)
        ).reset_index(drop=True)
        logger.info(f"Resampled dataset length: {len(self.df)}")

    def generate_train_test_dev_set(self, test_size: float = 0.15, dev_size: float = 0.15) -> None:
        logger.info("Generating train, test, and dev sets")
        if 'gold_label' not in self.df.columns:
            raise ValueError("DataFrame must contain 'gold_label' column for stratification.")

        remaining_size = test_size + dev_size
        if remaining_size >= 1.0:
            raise ValueError("The sum of test_size and dev_size must be less than 1.0")

        train_df, temp_df = train_test_split(
            self.df, test_size=remaining_size, random_state=42, stratify=self.df['gold_label']
        )
        relative_test_size = test_size / remaining_size
        test_df, dev_df = train_test_split(
            temp_df, test_size=relative_test_size, random_state=42, stratify=temp_df['gold_label']
        )

        self._save_split(train_df, 'train.csv')
        self._save_split(test_df, 'test.csv')
        self._save_split(dev_df, 'dev.csv')

        self.sub_dfs_train = train_df
        self.sub_dfs_test = test_df
        self.sub_dfs_dev = dev_df

        logger.info(
            f"Generated datasets - Train: {len(train_df)}, Test: {len(test_df)}, Dev: {len(dev_df)}"
        )

    def _save_split(self, df: pd.DataFrame, filename: str) -> None:
        save_path = os.path.join(self.workdir, filename)
        df.to_csv(save_path, index=False)
        logger.info(f"Saved {filename} to {save_path}")

    def validate(self) -> None:
        logger.info("Validation not yet implemented (NYI)")
