import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math
import logging
from typing import List, Optional, Union, Any

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PlottingUtils:
    def plot_columns(
        self,
        df: pd.DataFrame,
        target_column: str = "score",
        columns: Optional[List[str]] = None,
    ) -> None:
        if columns is None:
            columns = [
                'jsd', 'entropy_tag', 'entropy_all',
                'variance_tag', 'variance_all', 'highest_agreement'
            ]

        missing_columns = [col for col in [target_column] + columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Columns not found in DataFrame: {missing_columns}")

        num_plots = len(columns)
        num_rows = 2
        num_cols = math.ceil(num_plots / num_rows)
        plt.figure(figsize=(15, 10))

        for i, measure in enumerate(columns, 1):
            plt.subplot(num_rows, num_cols, i)
            sns.scatterplot(x=df[measure], y=df[target_column], alpha=0.7)
            sns.regplot(
                x=measure,
                y=target_column,
                data=df,
                scatter=False,
                color='red',
                line_kws={'linewidth': 1.5}
            )
            correlation = df[measure].corr(df[target_column])
            plt.title(f'{target_column} vs {measure}\nCorrelation = {correlation:.6f}')
            plt.xlabel(measure)
            plt.ylabel(target_column)
            plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.show()

    def show_percentile_comparison(
        self,
        workload: Any = None,
        workload_to_compare: Any = None,
        show_percentile: bool = False,
        percentile: float = 1.0,
        percentile_columns: Union[str, List[str]] = "score",
        columns: Optional[List[str]] = None,
    ) -> None:
        if columns is None:
            columns = ['score', 'highest_agreement', 'entropy_all', 'jsd']

        if workload is None and workload_to_compare is None:
            raise ValueError("At least one workload must be provided.")

        if workload_to_compare is not None:
            workload_to_compare.df['Model'] = workload_to_compare.name
        if workload is not None:
            workload.df['Model'] = workload.name

        if workload is not None and workload_to_compare is not None:
            df_combined = pd.concat([workload.df, workload_to_compare.df], ignore_index=True)
        else:
            df_combined = workload.df if workload else workload_to_compare.df

        required_columns = columns + (
            percentile_columns if isinstance(percentile_columns, list) else [percentile_columns]
        ) + ['Model']
        missing_columns = [col for col in required_columns if col not in df_combined.columns]
        if missing_columns:
            raise ValueError(f"Columns not found in DataFrame: {missing_columns}")

        percentiles = df_combined[columns].quantile(percentile)
        logger.info(f"Percentiles at {percentile * 100}%:\n{percentiles}")

        df_filtered = df_combined.copy()
        for col in (
            percentile_columns if isinstance(percentile_columns, list) else [percentile_columns]
        ):
            df_filtered = df_filtered[df_filtered[col] <= percentiles[col]]

        g = sns.PairGrid(df_filtered, hue='Model', vars=columns, diag_sharey=False)
        g.map_offdiag(sns.scatterplot, alpha=0.7)
        g.map_offdiag(sns.regplot, scatter=False, truncate=False)

        if show_percentile:
            for i, ax in enumerate(g.axes.flat):
                x_var = g.x_vars[i % len(g.x_vars)]
                y_var = g.y_vars[i // len(g.x_vars)]
                if x_var != y_var:
                    ax.axvline(
                        percentiles[x_var],
                        color='red',
                        linestyle='--',
                        label=f'{percentile * 100:.0f}th Percentile {x_var}'
                    )
                    ax.axhline(
                        percentiles[y_var],
                        color='red',
                        linestyle='--',
                        label=f'{percentile * 100:.0f}th Percentile {y_var}'
                    )

        g.map_diag(sns.kdeplot)
        g.add_legend(title="Model", bbox_to_anchor=(1.05, 0.95), loc="upper left")
        workload_name = workload.name if workload else "None"
        compare_name = workload_to_compare.name if workload_to_compare else "None"
        title = f'Comparison of {workload_name} and {compare_name}'
        if percentile < 1.0:
            title += f' (Filtered by {percentile * 100:.0f}th Percentile on {percentile_columns})'
        plt.suptitle(title, y=1.02, fontsize=16)
        plt.tight_layout()
        plt.show()

    def compare_correlations(
        self,
        workload1: Any,
        workload2: Any,
        base_columns: Optional[List[str]] = None,
        columns_to_compare: Optional[List[str]] = None,
    ):
        if base_columns is None:
            base_columns = ['score', 'highest_agreement']
        if columns_to_compare is None:
            columns_to_compare = ['score', 'highest_agreement', 'entropy_all', 'jsd', 'variance_tag']

        if workload1 is None or workload2 is None:
            raise ValueError("Both workload1 and workload2 must be provided.")

        comparison_data = []
        for base_col in base_columns:
            for column in columns_to_compare:
                if column != base_col and column in workload1.df.columns and column in workload2.df.columns:
                    if workload1.df[column].nunique() > 1 and workload2.df[column].nunique() > 1:
                        corr_wl1 = workload1.df[base_col].corr(workload1.df[column])
                        corr_wl2 = workload2.df[base_col].corr(workload2.df[column])
                        difference = corr_wl1 - corr_wl2
                        comparison_data.append({
                            'Base Column': base_col,
                            'Compared With': column,
                            f'Corr_{workload1.name}': corr_wl1,
                            f'Corr_{workload2.name}': corr_wl2,
                            'Difference': difference
                        })

        if not comparison_data:
            raise ValueError("No valid correlations could be computed.")

        comparison_table = pd.DataFrame(comparison_data)
        for col in [f'Corr_{workload1.name}', f'Corr_{workload2.name}', 'Difference']:
            comparison_table[col] = comparison_table[col].round(6)

        return comparison_table

    def compare_baseline_to_technique(
        self,
        workload_df: pd.DataFrame,
        workload_base_df: pd.DataFrame,
        columns_to_analyze: Optional[List[str]] = None,
        label1: str = "Workload",
        label2: str = "Workload Base",
        plot_results: bool = True,
    ) -> None:
        if columns_to_analyze is None:
            columns_to_analyze = [
                'score', 'highest_agreement', 'entropy_all', 'variance_all', 'jsd'
            ]

        mean_values = self._display_mean_values(workload_df, columns_to_analyze, label1)
        mean_values_base = self._display_mean_values(workload_base_df, columns_to_analyze, label2)
        if plot_results:
            self._plot_mean_comparison(mean_values, mean_values_base, label1, label2)

    def _display_mean_values(
        self,
        df: pd.DataFrame,
        columns: List[str],
        dataset_name: str = "Dataset",
    ) -> pd.Series:
        missing_columns = [col for col in columns if col not in df.columns]
        if missing_columns:
            raise ValueError(
                f"Columns not found in {dataset_name} DataFrame: {missing_columns}"
            )

        mean_values = df[columns].mean()
        logger.info(f"\nMean Values for {dataset_name}:\n{mean_values}")
        return mean_values

    def _plot_mean_comparison(
        self,
        mean1: pd.Series,
        mean2: pd.Series,
        label1: str = "Dataset 1",
        label2: str = "Dataset 2",
    ) -> None:
        comparison_df = pd.DataFrame({label1: mean1, label2: mean2})
        ax = comparison_df.plot(kind='bar', figsize=(10, 6), width=0.8)

        plt.title("Mean Value Comparison")
        plt.ylabel("Mean Value")
        plt.xticks(rotation=0)
        plt.grid(True, axis='y', linestyle='--', alpha=0.6)

        for p in ax.patches:
            height = p.get_height()
            ax.annotate(
                f"{height:.4f}",
                (p.get_x() + p.get_width() / 2., height),
                ha='center',
                va='center',
                xytext=(0, 9),
                textcoords='offset points',
                fontsize=10
            )

        plt.tight_layout()
        plt.show()
