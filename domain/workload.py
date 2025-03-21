import json
import os
import logging
from typing import List, Optional, Dict, Any, Tuple

import pandas as pd

from model_utils.model_utils import ModelUtils
from utils.ModelEvaluator_utils import ModelEvaluator
from utils.plotting_utils import PlottingUtils
from utils.ambi_utils import AmbiUtils
from model_utils.techniques.upper_bounds import Upper_bounds_trainer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Workload:
    def __init__(
        self,
        dataset_path: str,
        created_at: str,
        labels: List[str],
        name: str,
        source_path: str,
    ):
        self.dataset_path = dataset_path
        self.created_at = created_at
        self.labels = labels
        self.name = name
        self.source_path = source_path
        self.seeds: List[int] = []
        self.epochs: List[int] = []
        self.model_name = ""
        self.models: List[ModelUtils] = []
        self.df: Optional[pd.DataFrame] = None

    def to_json(self) -> str:
        return json.dumps(self, cls=WorkloadEncoder, indent=4)

    @classmethod
    def load(
        cls, workload_name: str = "instance_undefined", load_models: bool = True, quiet: bool = False
    ) -> "Workload":
        base_path = os.path.join(os.getcwd(), "saved_results", workload_name)
        save_df_path = os.path.join(base_path, "test.csv")
        save_results_path = os.path.join(base_path, "results.json")
        saved_models_path = os.path.join(os.getcwd(), "models")

        if not quiet:
            logger.info(f"Loading DataFrame from: {save_df_path}")
            logger.info(f"Loading results from: {save_results_path}")

        if not os.path.exists(save_df_path):
            raise FileNotFoundError(f"The DataFrame file {save_df_path} does not exist.")
        if not os.path.exists(save_results_path):
            raise FileNotFoundError(f"The results file {save_results_path} does not exist.")

        with open(save_results_path, 'r') as f:
            data = json.load(f)

        workload = cls(
            dataset_path=save_df_path,
            created_at=data['created_at'],
            labels=data['labels'],
            name=data['name'],
            source_path=save_results_path,
        )

        workload.seeds = data.get('seeds', [])
        workload.epochs = data.get('epochs', [])
        workload.model_name = data.get('model_name', "")
        workload.df = pd.read_csv(save_df_path)

        if load_models:
            for idx in range(len(workload.epochs)):
                model_instance = ModelUtils(workload.name + str(idx), workload.model_name, workload.labels)
                model_instance.load_model(os.path.join(saved_models_path, workload.name + str(idx)))
                workload.models.append(model_instance)

        return workload

    @classmethod
    def create(
        cls,
        name: str = "instance_undefined",
        model_name: str = 'bert-base-uncased',
        seeds: Optional[List[int]] = None,
        epochs: Optional[List[int]] = None,
    ) -> "Workload":
        seeds = seeds or [0]
        epochs = epochs or [1]

        work_dir = os.path.join(os.getcwd(), "workload")
        save_df_path = os.path.join(work_dir, "test.csv")
        struct_data_path = os.path.join(work_dir, 'struct_data.json')

        if not os.path.exists(struct_data_path):
            raise FileNotFoundError(f"The structure data file {struct_data_path} does not exist.")

        with open(struct_data_path, 'r') as f:
            data = json.load(f)

        workload = cls(
            dataset_path=data['dataset_path'],
            created_at=data['created_at'],
            labels=data['labels'],
            source_path=data['source_path'],
            name=name,
        )

        workload.seeds = seeds
        workload.epochs = epochs
        workload.model_name = model_name

        if os.path.exists(save_df_path):
            workload.df = pd.read_csv(save_df_path)
        else:
            logger.warning(f"Dataset path {save_df_path} does not exist.")

        for idx in range(len(epochs)):
            model_instance = ModelUtils(workload.name + str(idx), workload.model_name, workload.labels)
            workload.models.append(model_instance)

        logger.info(
            f"Created {workload.name} with model {workload.model_name}, epochs {workload.epochs}, and seeds {workload.seeds}"
        )
        return workload

    def train(self, smoothing: float = 0.0, dropout: Optional[float] = None, training_type: str = "ensemble"):
        for idx, model in enumerate(self.models):
            max_epochs = self.epochs[idx]
            seed = self.seeds[idx]

            if dropout is not None and training_type != "ub":
                logger.info(f"Training model {idx} with dropout={dropout}, smoothing={smoothing}")
                model.train_with_dropout(max_epochs=max_epochs, seed=seed, dropout=dropout, smoothing=smoothing)
            elif training_type != "ub":
                logger.info(f"Training model {idx} with smoothing={smoothing}")
                model.train(max_epochs=max_epochs, seed=seed, smoothing=smoothing)
            elif training_type == "ub":
                logger.info(f"Training model {idx} with upper_bounds")
                ub_trainer = Upper_bounds_trainer(instance_name=self.name, model_name=self.model_name)
                ub_trainer.train(max_epochs=max_epochs, seed=seed, smoothing=smoothing)
                # for now keep the model in the ub_trainer, so we can use it for evaluation later
                model.classifier = ub_trainer
                

    def evaluate(self, evaluation_type: str = "ensemble", debug: bool = False, num_samples: int = 100):
        logger.info(f"Requested evaluation type: {evaluation_type}. Starting evaluation...")
        evaluator = ModelEvaluator()

        if evaluation_type == "ensemble":
            evaluator.evaluate(workload=self, eval_type=evaluation_type, debug=debug)
        elif evaluation_type == "mc_dropout":
            evaluator.evaluate(workload=self, eval_type=evaluation_type, num_samples=num_samples, debug=debug)
        elif evaluation_type == "ub":
            evaluator.evaluate(workload=self, eval_type=evaluation_type, debug=debug)
        else:
            raise NotImplementedError(f"Evaluation type '{evaluation_type}' is not supported.")

    def save(self):
        base_path = os.path.join(os.getcwd(), "saved_results", self.name)
        save_df_path = os.path.join(base_path, "test.csv")
        save_results_path = os.path.join(base_path, "results.json")

        logger.info(f"Saving DataFrame to {save_df_path}")
        logger.info(f"Saving results to {save_results_path}")

        os.makedirs(base_path, exist_ok=True)

        if self.df is not None:
            self.df.to_csv(save_df_path, index=False)
        else:
            logger.warning("DataFrame is empty. Nothing to save.")

        with open(save_results_path, 'w') as f:
            f.write(self.to_json())

        logger.info(f"Results saved to {save_results_path}")

    def info(self):
        logger.info(f"Workload name: {self.name}")
        logger.info(f"Created at: {self.created_at}")
        logger.info(f"Labels: {self.labels}")
        logger.info(f"Source path: {self.source_path}")
        logger.info(f"Dataset path: {self.dataset_path}")
        logger.info(f"Model name: {self.model_name}")
        logger.info(f"Seeds: {self.seeds}")
        logger.info(f"Epochs: {self.epochs}")
        logger.info(f"Models: {self.models}")

    def generate_plots(self):
        plot_utils = PlottingUtils()
        plot_utils.generate_plots(workload=self)

    def compare_baseline_to_technique(
        self,
        workload_base: 'Workload',
        percentile: float = 100.0,
        columns_to_analyze: Optional[List[str]] = None,
        label1: str = "Workload",
        label2: str = "Workload Base",
        plot_results: bool = True,
    ):
        if columns_to_analyze is None:
            columns_to_analyze = ['score', 'highest_agreement', 'entropy_all', 'variance_all', 'jsd']

        df_filtered_wl, df_filtered_base = self._apply_percentile(self.df, workload_base.df, percentile)
        plot_utils = PlottingUtils()
        plot_utils.compare_baseline_to_technique(
            df_filtered_wl, df_filtered_base, columns_to_analyze, label1, label2, plot_results
        )

    def plot_columns(self, target_column: str = "score", columns: Optional[List[str]] = None):
        if columns is None:
            columns = ['jsd', 'entropy_tag', 'entropy_all', 'variance_tag', 'variance_all', 'highest_agreement']
        plot_utils = PlottingUtils()
        plot_utils.plot_columns(df=self.df, target_column=target_column, columns=columns)

    def show_percentile_comparison(
        self,
        workload_to_compare: Optional['Workload'] = None,
        show_percentile: bool = False,
        percentile: float = 100.0,
        percentile_columns: str = "score",
        columns: Optional[List[str]] = None,
    ):
        if columns is None:
            columns = ['score', 'highest_agreement', 'entropy_all', 'jsd']

        plot_utils = PlottingUtils()
        plot_utils.show_percentile_comparison(
            workload=self,
            workload_to_compare=workload_to_compare,
            show_percentile=show_percentile,
            percentile=percentile,
            percentile_columns=percentile_columns,
            columns=columns,
        )

    def compare_correlations(
        self,
        workload_to_compare: Optional['Workload'] = None,
        base_columns: Optional[List[str]] = None,
        columns_to_compare: Optional[List[str]] = None,
    ) -> Any:
        if base_columns is None:
            base_columns = ['score', 'highest_agreement']
        if columns_to_compare is None:
            columns_to_compare = ['score', 'highest_agreement', 'entropy_all', 'jsd', 'variance_tag']

        plot_utils = PlottingUtils()
        return plot_utils.compare_correlations(
            workload1=self,
            workload2=workload_to_compare,
            base_columns=base_columns,
            columns_to_compare=columns_to_compare,
        )

    def threshold_based_ambiguity_detection(
        self,
        columns: Optional[List[str]] = None,
        threshold_percentiles: Optional[Dict[str, float]] = None,
        default_threshold_percentile: float = 90.0,
        debug: bool = False,
    ):
        if columns is None:
            columns = ["variance_tag", "entropy_all", "jsd"]

        ambi_utils = AmbiUtils()
        ambi_utils.threshold_based_ambiguity_detection(
            workload=self.df,
            columns=columns,
            threshold_percentiles=threshold_percentiles,
            default_threshold_percentile=default_threshold_percentile,
            debug=debug,
        )

    def agreement_based_ambiguity_detection(
        self,
        agreement_column: str = "highest_agreement",
        threshold_percentile: float = 60.0,
        debug: bool = False,
    ):
        ambi_utils = AmbiUtils()
        ambi_utils.agreement_based_ambiguity_detection(
            df=self.df,
            agreement_column=agreement_column,
            threshold_percentile=threshold_percentile,
            debug=debug,
        )

    def test_detection(
        self,
        columns: Optional[List[str]] = None,
        threshold_percentiles: Optional[Dict[str, float]] = None,
        default_threshold_percentile: float = 90.0,
        agreement_column: str = "highest_agreement",
        threshold_percentile_agreement: float = 60.0,
        debug: bool = False,
        quiet: bool = False,
        random_threshold: float = None
    ):
        if columns is None:
            columns = ["variance_tag", "entropy_all", "jsd"]

        ambi_utils = AmbiUtils()
        ambi_utils.test_detection(
            workload=self.df,
            columns=columns,
            threshold_percentiles=threshold_percentiles,
            default_threshold_percentile=default_threshold_percentile,
            agreement_column=agreement_column,
            threshold_percentile_agreement=threshold_percentile_agreement,
            debug=debug,
            quiet=quiet,
            random_threshold=random_threshold
        )

    def find_best_threshold_combination_based_on_true_matches(
        self,
        columns: Optional[List[str]] = None,
        min_threshold: int = 60,
        max_threshold: int = 99,
        min_agreement_threshold: int = 70,
        max_agreement_threshold: int = 70,
        agreement_column: str = 'highest_agreement',
        debug: bool = False,
    ):
        if columns is None:
            columns = ["variance_tag", "entropy_all", "jsd"]

        ambi_utils = AmbiUtils()
        ambi_utils.find_best_threshold_combination_based_on_true_matches(
            workload=self.df,
            columns=columns,
            min_threshold=min_threshold,
            max_threshold=max_threshold,
            min_agreement_threshold=min_agreement_threshold,
            max_agreement_threshold=max_agreement_threshold,
            agreement_column=agreement_column,
            debug=debug,
        )

    def find_best_threshold_combination_based_on_error_rate(
        self,
        columns: Optional[List[str]] = None,
        min_threshold: int = 60,
        max_threshold: int = 99,
        min_agreement_threshold: int = 70,
        max_agreement_threshold: int = 70,
        agreement_column: str = 'highest_agreement',
        debug: bool = False,
    ):
        if columns is None:
            columns = ["variance_tag", "entropy_all", "jsd"]

        ambi_utils = AmbiUtils()
        return ambi_utils.find_best_threshold_combination_based_on_error_rate(
            workload=self.df,
            columns=columns,
            min_threshold=min_threshold,
            max_threshold=max_threshold,
            min_agreement_threshold=min_agreement_threshold,
            max_agreement_threshold=max_agreement_threshold,
            agreement_column=agreement_column,
            debug=debug,
        )

    def show_confusion_matrix(
        self,
        y_true: str = 'ambiguous_based_on_agreement',
        y_pred: str = 'ambiguous',
        plot: bool = True,
    ):
        ambi_utils = AmbiUtils()
        ambi_utils.show_confusion_matrix(
            workload=self.df, y_true_col=y_true, y_pred_col=y_pred, plot=plot
        )

    def show_eval_metrics_acc_recall_prec_f1(
        self,
        y_true: str = 'ambiguous_based_on_agreement',
        y_pred: str = 'ambiguous',
    ):
        ambi_utils = AmbiUtils()
        return ambi_utils.show_eval_metrics(
            workload=self.df, y_true_col=y_true, y_pred_col=y_pred
        )

    def show_roc_auc_curve(
        self,
        y_true: str = 'ambiguous_based_on_agreement',
        predictors: Optional[List[str]] = None,
        plot: bool = True,
    ):
        if predictors is None:
            predictors = ['entropy_all', 'jsd', 'variance_tag', 'score']

        ambi_utils = AmbiUtils()
        ambi_utils.show_roc_auc_curve(
            workload=self.df, y_true_col=y_true, predictors=predictors, plot=plot
        )

    def show_combined_auc(
        self,
        y: str = 'ambiguous_based_on_agreement',
        predictors: Optional[List[str]] = None,
    ):
        if predictors is None:
            predictors = ['entropy_all', 'jsd', 'variance_tag', 'score']

        ambi_utils = AmbiUtils()
        ambi_utils.show_combined_auc(workload=self.df, y_col=y, predictors=predictors)

    def calculate_optimal_predictor_threshold(
        self,
        y: str = 'ambiguous_based_on_agreement',
        predictors: Optional[List[str]] = None,
    ):
        if predictors is None:
            predictors = ['entropy_all', 'jsd', 'variance_tag', 'score']

        ambi_utils = AmbiUtils()
        ambi_utils.calculate_optimal_predictor_threshold(
            workload=self.df, y_col=y, predictors=predictors
        )

    @staticmethod
    def _apply_percentile(
        df_wl: pd.DataFrame, df_base: pd.DataFrame, percentile: float = 100.0
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        if percentile < 100.0:
            columns = ['score', 'highest_agreement', 'entropy_all']
            quantile_value = percentile / 100.0
            percentiles_wl = df_wl[columns].quantile(quantile_value)
            percentiles_base = df_base[columns].quantile(quantile_value)

            df_filtered_wl = df_wl[(df_wl[columns] <= percentiles_wl).all(axis=1)]
            df_filtered_base = df_base[(df_base[columns] <= percentiles_base).all(axis=1)]
        else:
            df_filtered_wl = df_wl.copy()
            df_filtered_base = df_base.copy()

        return df_filtered_wl, df_filtered_base


class WorkloadEncoder(json.JSONEncoder):
    def default(self, obj: Any) -> Any:
        if isinstance(obj, Workload):
            return {
                'dataset_path': obj.dataset_path,
                'created_at': obj.created_at,
                'labels': obj.labels,
                'name': obj.name,
                'source_path': obj.source_path,
                'seeds': obj.seeds,
                'epochs': obj.epochs,
                'model_name': obj.model_name,
            }
        return super().default(obj)
