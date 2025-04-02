import os
import logging
from typing import List, Dict

import pandas as pd
import numpy as np
from scipy.stats import entropy
from scipy.spatial.distance import jensenshannon
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    accuracy_score,
    roc_curve,
    roc_auc_score,
    auc,
)

import matplotlib.pyplot as plt
import seaborn as sns
import math

from domain.workload import Workload

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
file_handler = logging.FileHandler('./Logs/workload_evaluator.log')
console_handler.setLevel(logging.INFO)
file_handler.setLevel(logging.DEBUG)

formatter = logging.Formatter('%(asctime)s %(levelname)s:%(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)




class WorkloadEvaluator:
    def __init__(
        self,
        models: List[str],
        datasets: List[str],
        techniques: List[str],
        num_runs: int = 3,
        enable_plotting: bool = True,
    ):

        self.models = models
        self.datasets = datasets
        self.techniques = techniques
        self.num_runs = num_runs
        self.enable_plotting = enable_plotting
        self.workloads = {}
        self.results = []
        self.label_columns = []
        self.analysis_results_dir = 'analysis_results'
        self.summary_dir = os.path.join(self.analysis_results_dir, 'summary_tables')
        self.plot_dir = os.path.join(self.analysis_results_dir, 'plots')
        self.latex_dir = os.path.join(self.analysis_results_dir, 'latex')
        self.markdown_dir = os.path.join(self.analysis_results_dir, 'markdown')
        os.makedirs(self.analysis_results_dir, exist_ok=True)
        os.makedirs(self.summary_dir, exist_ok=True)
        os.makedirs(self.latex_dir, exist_ok=True)
        os.makedirs(self.plot_dir, exist_ok=True)
        os.makedirs(self.markdown_dir, exist_ok=True)
        self.mapping_helper = {
            'roberta': 'RoBERTa',
            'bert': 'BERT',
            'xlnet': 'XLNet',
            'rt': 'Rotten Tomatoes',
            'go_emotions': 'GoEmotions',
            'hate_gap': 'GAB Hate Speech',
            'baseline': 'Baseline',
            'mc': 'MC Dropout',
            'smoothing': 'Label Smoothing',
            'de': 'Deep Ensemble',
            'random': 'Random Baseline',
            'prajjwal1/bert-small' :'prajjwal1-bert-small',
            'ub': 'Oracle',
            'entropy_agreement': 'Label Ambiguity Score',
            'entropy_mean_prediction': 'Model Uncertainty Score',
            'Baseline': 'Baseli.',
            'MC Dropout': 'MCD',
            'Deep Ensemble': 'DE',
            "Label Smoothing": "LS",
            'Upper Bound': 'Oracle',
            'Oracle': 'Oracle', #  Oracle Fine-Tuning
        }
        
        

    ## Helper Functions

    @staticmethod
    def load_labels(workload: Workload) -> List[str]:

        label_columns = [col for col in workload.df.columns if col.endswith('_probability')]
        return label_columns

    @staticmethod
    def compute_entropy(row, label_columns):

        probabilities = row[label_columns].astype(float).values

        probabilities[probabilities < 0] = 0.0

        #this is needed: normalize probabilities to sum to 1
        total = probabilities.sum()
        if total > 0:
            probabilities /= total
        else:
            probabilities = np.ones_like(probabilities) / len(probabilities)

        return entropy(probabilities, base=2)

    @staticmethod
    def compute_entropy_invert(row, label_columns):
        entropy_value = WorkloadEvaluator.compute_entropy(row, label_columns)
        return 1 - entropy_value


    
    ### Implementations
    def calculate_evaluation_metrics_for_base(self,column_filter = None, print_latex=False):
        logger.info("Calculating evaluation metrics...")
        results = []
        for model in models:
            for ds in datasets:
                baseline_metrics = None 
                
                for tech in techniques:
                    precision_scores = []
                    recall_scores = []
                    f1_scores = []
                    accuracy_scores = []
                    
                    for run in range(1, num_runs + 1):
                        workload_name = f"{model}_{ds}_{tech}_{run}"
                        workload = Workload.load(workload_name=workload_name, load_models=False, quiet=True)
                        
                        y_true = workload.df['gold_label']
                        y_pred = workload.df['tag']
                        
                        precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
                        recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
                        f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
                        accuracy = accuracy_score(y_true, y_pred)
                        
                        precision_scores.append(precision)
                        recall_scores.append(recall)
                        f1_scores.append(f1)
                        accuracy_scores.append(accuracy)
                    
                    precision_mean = pd.Series(precision_scores).mean()
                    precision_std = pd.Series(precision_scores).std()
                    recall_mean = pd.Series(recall_scores).mean()
                    recall_std = pd.Series(recall_scores).std()
                    f1_mean = pd.Series(f1_scores).mean()
                    f1_std = pd.Series(f1_scores).std()
                    accuracy_mean = pd.Series(accuracy_scores).mean()
                    accuracy_std = pd.Series(accuracy_scores).std()
                    
                    #calculate improvement over baseline
                    if tech == "baseline":
                        improvement_f1 = 0  
                        baseline_metrics = {
                            'Precision': precision_mean,
                            'Recall': recall_mean,
                            'F1 Score': f1_mean,
                            'Accuracy': accuracy_mean
                        }
                    else:
                        if baseline_metrics:
                            improvement_f1 = f1_mean - baseline_metrics['F1 Score']
                        else:
                            improvement_f1 = None 
                    
                    results.append({
                        'Model': model,
                        'Dataset': ds,
                        'Technique': tech,
                        'Mean Precision': round(precision_mean, 4),
                        'Std Precision': round(precision_std, 4),
                        'Mean Recall': round(recall_mean, 4),
                        'Std Recall': round(recall_std, 4),
                        'Mean F1 Score': round(f1_mean, 4),
                        'Std F1 Score': round(f1_std, 4),
                        'Mean Accuracy': round(accuracy_mean, 4),
                        'Std Accuracy': round(accuracy_std, 4),
                        'Improvement over Baseline (F1)': round(improvement_f1, 4) if improvement_f1 is not None else '-'
                    })

        results_df = pd.DataFrame(results)
        # Ffilter columns if filter is present
        if column_filter is not None:
            results_df = results_df[column_filter]
        #results_df = results_df[[
        #    'Model', 'Dataset', 
        #    'Mean Precision',
        #    'Mean Recall', 
        #    'Mean F1 Score', 
        #    'Mean Accuracy', 
        #]]


        # Generate LaTeX table
        latex_table = results_df.to_latex(
            index=False,
            float_format="%.2f",
            na_rep='-',
            longtable=True,
            caption="Evaluation Metrics for Models, Datasets, and Techniques",
            label="tab:evaluation_metrics"
        )
        if print_latex:
            print(latex_table)
        latex_path = os.path.join(self.latex_dir, 'evaluation_metrics_for_base.tex')
        with open(latex_path, 'w') as f:
            f.write(latex_table)
        logger.info(f"LaTeX correlation table saved to {latex_path}")
        
        #save MD
        md_path = os.path.join(self.markdown_dir, 'evaluation_metrics_for_base.md')
        with open(md_path, 'w') as f:
            f.write(results_df.to_markdown())
        logger.info(f"Markdown table saved to {latex_path}")



    def generate_latex_multiro_table(self, long_table, caption, label):
        def latex_escape(text):
            if not isinstance(text, str):
                text = str(text)
            text = text.replace("%", "\\%")
            text = text.replace("+-", "$\\pm$")
            text = text.replace("±", "$\\pm$")
            return text

        
        datasets = long_table["Dataset"].unique()
        pivoted_rows = []
        for (model, technique), group in long_table.groupby(["Model", "Technique"]):
            technique = self.mapping_helper[technique] # shorten the technique name
            row = {"Model": model, "Method": technique}
            mean_vals = []
            for ds in datasets:
                sub = group[group["Dataset"] == ds]
                if not sub.empty:
                    data = sub.iloc[0]
                    mean_value = data['Mean Correlation']
                    if isinstance(mean_value, str):
                        # it's only string when \\textbf is added
                        combined_corr = f"{data['Mean Correlation']} $\\pm$ {data['Std Dev %']:.3f}"
                    else:
                        # fix, make sure all have the same number of decimal places when printed out
                        combined_corr = f"{data['Mean Correlation']:.3f} $\\pm$ {data['Std Dev %']:.3f}"
                        
                    row[f"{ds} Corr."] = combined_corr
                    row[f"{ds} % Improv."] = data["Imp. over Baseline"]
                    try:
                        mean_vals.append(float(data["Mean Correlation"]))
                    except Exception:
                        pass
                else:
                    row[f"{ds} Corr."] = ""
                    row[f"{ds} % Improv."] = ""
            if mean_vals:
                row["Average Corr."] = round(sum(mean_vals) / len(mean_vals), 3)
            else:
                row["Average Corr."] = ""
            pivoted_rows.append(row)

        wide_table = pd.DataFrame(pivoted_rows)
        dataset_cols = []
        for ds in datasets:
            dataset_cols.extend([f"{ds} Corr.", f"{ds} % Improv."])
        new_columns = ["Model", "Method"] + dataset_cols + ["Average Corr."]
        wide_table = wide_table[new_columns]

        n_pairs = len(datasets)
        total_cols = 2 + 2 * n_pairs + 1
        col_format = "ll" + "c" * (total_cols - 2)

        header  = "\\begin{table*}[h!]\n"
        header += "\\setlength{\\tabcolsep}{4pt}\n"
        header += "\\centering\n"
        header += "\\small\n\n"
        header += "\\begin{tabular}{" + col_format + "}\n"
        header += "\\toprule\n"

        first_row = "\\multicolumn{1}{c}{} & \\multicolumn{1}{c}{}"
        for ds in datasets:
            first_row += " & \\multicolumn{2}{c}{\\textbf{" + ds + "}}"
        first_row += " & \\multicolumn{1}{c}{\\textbf{Average}} \\\\ \n"
        second_row = "\\textbf{Model} & \\textbf{Method}"
        for ds in datasets:
            second_row += " & \\textbf{Corr.} & \\textbf{\\% Improv.}"
        second_row += " & \\textbf{Corr.} \\\\ \n"

        header += first_row + second_row
        header += "\\midrule\n"

        body = ""
        model_groups = list(wide_table.groupby("Model"))
        for idx, (model, group) in enumerate(model_groups):
            group = group.reset_index(drop=True)
            nrows = len(group)
            for i, row in group.iterrows():
                if i == 0:
                    model_cell = "\\multirow{" + str(nrows) + "}{*}{" + latex_escape(row["Model"]) + "}"
                else:
                    model_cell = ""
                if row['Method'] == 'Oracle':
                    body += " \cmidrule{2-9}"
                row_cells = [model_cell, latex_escape(row["Method"])]
                for ds in datasets:
                    row_cells.append(latex_escape(row[f"{ds} Corr."]))
                    row_cells.append(latex_escape(row[f"{ds} % Improv."]))
                row_cells.append(latex_escape(row["Average Corr."]))
                row_line = " & ".join(row_cells) + " \\\\ \n"
                body += row_line
            if idx < len(model_groups) - 1:
                body += "\\midrule\n"
            
        footer  = "\\bottomrule\n"
        footer += "\\end{tabular}\n"
        footer += "\\caption{" + caption + "}\n"
        footer += "\\label{" + label + "}\n"
        footer += "\\end{table*}"

        return header + body + footer

    
    def ambiguity_human_vs_models_correlation(self):
        latex_table_prepared =pd.DataFrame(columns=['Technique','Dataset','Model' ,'Mean Correlation','Std Dev %','Imp. over Baseline'])
        print("models", self.models, "datasets", self.datasets, "techniques", techniques)
        for model in self.models:
            for ds in self.datasets:
                for tech in techniques:
                    tabels = []
                    for i in range(0,3):
                        idx = (i % 3) + 1
                        current_evaluation = f"{model}_{ds}_{tech}_{idx}"
                        #current_evaluation = f"{model}_{ds}_{tech}_{idx +1}"
                        workload = Workload.load(workload_name = current_evaluation,load_models=False, quiet=True)
                        self.label_columns= self.load_labels(workload=workload)
                        workload.df['entropy_agreement'] = workload.df.apply(lambda row: self.compute_entropy(row, self.label_columns), axis=1) 
                        corr_table = workload.compare_correlations(workload_to_compare=workload,base_columns=['entropy_agreement'], columns_to_compare=['highest_agreement', 'entropy_agreement','entropy_mean_prediction', 'mean_variance', 'mean_jsd'])
                        tabels.append(corr_table)

                    correlation_series = []
                    for idx, table in enumerate(tabels, start=1):
                        corr_col_name = table.columns[2] 
                        correlation_series.append(table[corr_col_name])

                    correlation_df = pd.concat(correlation_series, axis=1)
                    correlation_df.columns = [f'Corr_{idx}' for idx in range(1, len(tabels) + 1)]

                    correlation_df['Mean Correlation'] = correlation_df.mean(axis=1)
                    correlation_df['Std Correlation'] = correlation_df.std(axis=1)

                    mean_std_corr_df = pd.DataFrame({
                        'Base Column': tabels[0]['Base Column'],
                        'Compared With': tabels[0]['Compared With'],
                        'Mean Correlation': correlation_df['Mean Correlation'],
                        'Std Correlation': correlation_df['Std Correlation']
                    })

                    for idx in range(1, len(tabels) + 1):
                        diff_col_name = f'Diff_Corr_{idx}'
                        mean_std_corr_df[diff_col_name] = correlation_df[f'Corr_{idx}'] - correlation_df['Mean Correlation']

                    mean_std_corr_df['Mean Correlation'] = mean_std_corr_df['Mean Correlation'].round(3)
                    mean_std_corr_df['Std Correlation'] = mean_std_corr_df['Std Correlation'].round(3)
                    for idx in range(1, len(tabels) + 1):
                        diff_col_name = f'Diff_Corr_{idx}'
                        mean_std_corr_df[diff_col_name] = mean_std_corr_df[diff_col_name].round(3)

                    
                    #base_value = 1
                    if tech.lower() == "baseline":
                        base_value = mean_std_corr_df['Mean Correlation'][1] # 1 is entropy
                    
                    if (mean_std_corr_df['Mean Correlation'][1] / base_value) < 1:
                        coeff = -1
                    else:
                        coeff = 1
                        
                    #TODO if base_value does not exist, we have to set it to some value. this may happen if we dont use a baseline technique
                    
                    if model in self.mapping_helper:
                        model_name = self.mapping_helper[model]
                    else:
                        model_name = model
                        
                    improvement = coeff * ( mean_std_corr_df['Mean Correlation'][1] / base_value - 1) * 100
                    improvement_str = f"{improvement:.0f}%"
                    
                    latex_table_prepared.loc[len(latex_table_prepared)] = [self.mapping_helper[tech],self.mapping_helper[ds], model_name, mean_std_corr_df['Mean Correlation'][1], mean_std_corr_df['Std Correlation'][1], improvement_str] 

                    latex_table = latex_table_prepared[['Technique','Dataset','Model','Mean Correlation','Std Dev %','Imp. over Baseline']].to_latex(index=False, 
                                        caption=f'Mean Correlation Coefficients between Uncertainty Metrics and Empirical Entropy (Average of 3 Runs)',
                                        label='table:roberta_rotten_tomatoes_correlations',
                                        column_format='lllcr',
                                        escape=True)
                    md_table_prepared = latex_table_prepared[['Technique','Dataset','Model','Mean Correlation','Std Dev %','Imp. over Baseline']]

                    #print(latex_table)

        #print(latex_table)
        latex_path = os.path.join(self.latex_dir, 'human_vs_models_correlatio.tex')
        with open(latex_path, 'w') as f:
            f.write(latex_table)
        logger.info(f"LaTeX correlation table saved to {latex_path}")
        
        #save MD
        md_path = os.path.join(self.markdown_dir, 'human_vs_models_correlatio.md')
        with open(md_path, 'w') as f:
            f.write(md_table_prepared.to_markdown())
        logger.info(f"Markdown table saved to {md_path}")
        
        
        
        #print(latex_table_prepared)
        all_dfs = []
        # The values are in a column mixed by model and dataset, therefore we have to seperate them by dataset and model, to find the max values to mark
        # also we only want to bold 'Mean Correlation', 'Imp. over Baseline', here we can use the parameter value_cols
        for model in self.models:
            for dataset in self.datasets:
                df_filtered = latex_table_prepared[
                    (latex_table_prepared['Model'] == self.mapping_helper[model]) & (latex_table_prepared['Dataset'] == self.mapping_helper[dataset])
                ].copy()
                if not df_filtered.empty:
                    df_filtered = self.bold_extreme_values(df_filtered, extreme='max', value_cols=['Mean Correlation', 'Imp. over Baseline'])
                    all_dfs.append(df_filtered)
        latex_table_prepared = pd.concat(all_dfs, ignore_index=True)
        #print(latex_table_prepared)
        
        
        
        improved_human_vs_model_latex = self.generate_latex_multiro_table(
            latex_table_prepared,
            caption=('Mean correlation coefficients for all models (mean $\pm$ std) and percentage improvement over baseline. '
                    'Bolded are the highest scores in each column per model.'),
            label='table:results_correlations'
        )
        new_latex_path = os.path.join(self.latex_dir, 'improved_human_vs_models_correlatio.tex')
        with open(new_latex_path, 'w') as f:
            f.write(improved_human_vs_model_latex)
        logger.info(f"New LaTeX correlation table saved to {new_latex_path}")


    def scatter_plot_correlation_user_vs_models_entropy(self):
        for model in self.models:
            for ds in self.datasets:
                for tech in techniques:
                    tabels = []
                    for i in range(0,3):
                        idx = (i % 3) + 1
                        current_evaluation = f"{model}_{ds}_{tech}_{idx}"
                        workload = Workload.load(workload_name = current_evaluation,load_models=False, quiet=True)
                        self.label_columns= self.load_labels(workload=workload)
                        workload.df['entropy_agreement'] = workload.df.apply(lambda row: self.compute_entropy(row, self.label_columns), axis=1) 
                        tabels.append(workload.df)

                    tabels[0]["entropy_agreement"] = (tabels[0]["entropy_agreement"] + tabels[1]["entropy_agreement"] + tabels[2]["entropy_agreement"]) / 3
                    tabels[0]["entropy_mean_prediction"] = (tabels[0]["entropy_mean_prediction"] + tabels[1]["entropy_mean_prediction"] + tabels[2]["entropy_mean_prediction"]) / 3
                    #print("tabels[0][entropy_agreement].mean()", tabels[0]["entropy_agreement"].corr(tabels[0]["entropy_mean_prediction"]))
                    mean_df = tabels[0]
                    workload.df = mean_df
                    #workload.plot_columns('entropy_mean_prediction', ['entropy_agreement']) 
                    # Generate a custom plot and not use the workload.plot_columns ... 
                    num_plots = len(['entropy_agreement'])
                    num_rows = 1
                    num_cols = math.ceil(num_plots / num_rows)
                    plt.figure(figsize=(8,3.3))
                    ent = 'entropy_mean_prediction'
                    for i, measure in enumerate(['entropy_agreement'], 1):
                        plt.subplot(num_rows, num_cols, i)
                        sns.scatterplot(x=workload.df[measure], y=workload.df['entropy_mean_prediction'], alpha=0.7)
                        sns.regplot(
                            x=measure,
                            y='entropy_mean_prediction',
                            data=workload.df,
                            scatter=False,
                            color='red',
                            line_kws={'linewidth': 1.5}
                        )
                        correlation = workload.df[measure].corr(workload.df['entropy_mean_prediction'])
                        #plt.title(f'Correlation = {correlation:.6f}')
                        plt.xlabel(self.mapping_helper[measure])
                        plt.ylabel(self.mapping_helper['entropy_mean_prediction'])
                        plt.grid(True, linestyle='--', alpha=0.6)
                    plt.tight_layout()
                    #plt.show()
                    model_path_fix = model.replace("/", "-")
                    plt.savefig(os.path.join(self.plot_dir, f'scatter_plot_correlation_user_vs_models_entropy_{model_path_fix}_{ds}_{tech}.png'))
                    plt.close()
                    
                    

    def scatter_plot_correlation_user_vs_models_entropy_combined(self):
        columns = ['entropy_agreement']  
        target_column = 'entropy_mean_prediction'  

        for ds in self.datasets:
            num_techs = len(techniques)
            num_cols = 2
            num_rows = math.ceil(num_techs / num_cols)
            fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 10))
            axes = axes.flatten()

            for idx_outer, tech in enumerate(techniques):
                ax = axes[idx_outer]
                all_data = []

                for model in self.models:
                    model_data = []

                    for idx in range(3): 
                        current_evaluation = f"{model}_{ds}_{tech}_{idx + 1}"
                        workload = Workload.load(workload_name=current_evaluation, load_models=False, quiet=True)
                        workload.df['Model'] = model
                        workload.df['Run'] = idx
                        self.label_columns= self.load_labels(workload)
                        workload.df['entropy_agreement'] = workload.df.apply(lambda row: self.compute_entropy(row, self.label_columns), axis=1) 
                        model_data.append(workload.df)

                    model_df = pd.concat(model_data, ignore_index=True)
                    all_data.append(model_df)

                df_combined = pd.concat(all_data, ignore_index=True)

                # Plotting
                for measure in columns:
                    sns.scatterplot(
                        data=df_combined,
                        x=measure,
                        y=target_column,
                        hue='Model',
                        alpha=0.7,
                        ax=ax
                    )
                    palette = sns.color_palette()
                    for idx_model, model in enumerate(self.models):
                        model_data = df_combined[df_combined['Model'] == model]
                        sns.regplot(
                            data=model_data,
                            x=measure,
                            y=target_column,
                            scatter=False,
                            ax=ax,
                            color=palette[idx_model],
                            line_kws={'linewidth': 1.5}
                        )

                    correlations = df_combined.groupby('Model').apply(lambda x: x[measure].corr(x[target_column]))
                    # Format correlation text
                    corr_text = '\n'.join([f"{model}: {corr:.4f}" for model, corr in correlations.items()])

                    ax.set_title(f'Tech: {self.mapping_helper[tech]} - Dataset: {self.mapping_helper[ds]}\nCorrelations:\n{corr_text}')
                    ax.set_xlabel(measure)
                    ax.set_ylabel(target_column)
                    ax.grid(True, linestyle='--', alpha=0.6)
                ax.legend()
            plt.tight_layout()
            model_path_fix = model.replace("/", "-")
            plt.savefig(os.path.join(self.plot_dir, f'scatter_plot_correlation_user_vs_models_entropy_combined_{model_path_fix}_{ds}_{tech}_{idx + 1}.png'))
            #plt.show()    
            plt.close()   


#TODO here we have to calc the std differently
    def calculate_JSD_MSE_CORR(self):
        
        latex_table_helper_list= []
        for ds in self.datasets:
            num_techs = len(techniques)
            num_cols = 2
            num_rows = math.ceil(num_techs / num_cols)
            fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 10))
            axes = axes.flatten()
            
            for idx_outer, tech in enumerate(techniques):
                
                for model in self.models:
                    model_data = []
                    tabels = []
                    for idx in range(3): 
                        current_evaluation = f"{model}_{ds}_{tech}_{idx + 1}"
                        workload = Workload.load(workload_name=current_evaluation, load_models=False, quiet=True)
                        self.label_columns= self.load_labels(workload)
                        workload.df['entropy_agreement'] = workload.df.apply(lambda row: self.compute_entropy(row, self.label_columns), axis=1) 
                        tabels.append(workload.df)

                        annotator_cols = [col for col in workload.df.columns if col.endswith('_probability')]
                        model_cols = [col for col in workload.df.columns if col.endswith('_probability_model')]
                        
                        # calculate JSD, MSE, CORR and the other metrics for each run separately
                        df = workload.df

                        probability_cols = annotator_cols + model_cols

                        for col in probability_cols:
                            df[col] = pd.to_numeric(df[col], errors='coerce')

                        df[probability_cols] = df[probability_cols].astype(np.float64)

                        # Check for NaNs
                        nan_counts = df[probability_cols].isna().sum()
                        #print("NaN counts in probability columns:\n", nan_counts)
                        epsilon = 1e-10
                        df[probability_cols] = df[probability_cols].fillna(epsilon)


                        #probabilities must sum to 1
                        df['annotator_total'] = df[annotator_cols].sum(axis=1)
                        df['model_total'] = df[model_cols].sum(axis=1)

                        df[annotator_cols] = df[annotator_cols].div(df['annotator_total'], axis=0)
                        df[model_cols] = df[model_cols].div(df['model_total'], axis=0)

                        from scipy.spatial.distance import jensenshannon

                        def compute_jsd(row):
                            p = row[annotator_cols].values.astype(np.float64)
                            q = row[model_cols].values.astype(np.float64)
                            p /= p.sum()
                            q /= q.sum()
                            js_distance = jensenshannon(p, q, base=2)
                            js_divergence = js_distance ** 2 
                            return js_divergence

                        df['JSD'] = df.apply(compute_jsd, axis=1)


                        def compute_correlation(row):
                            p = row[annotator_cols].values.astype(np.float64)
                            q = row[model_cols].values.astype(np.float64)
                            if np.std(p) == 0 or np.std(q) == 0:
                                return np.nan 
                            else:
                                corr = np.corrcoef(p, q)[0, 1]
                                #print("p:", p, "q:", q, "corr:", corr)
                                return corr


                        df['Correlation'] = df.apply(compute_correlation, axis=1)

                        def compute_kl_divergence(row):
                            p = row[annotator_cols].values.astype(np.float64)
                            q = row[model_cols].values.astype(np.float64)
                            #No division by zero or log(0)
                            epsilon = 1e-10
                            p = p + epsilon
                            q = q + epsilon
                            p /= p.sum()
                            q /= q.sum()
                            return entropy(p, qk=q)


                        df['KL_Divergence'] = df.apply(compute_kl_divergence, axis=1)

                        from sklearn.metrics.pairwise import cosine_similarity

                        def compute_cosine_similarity(row):
                            p = row[annotator_cols].values.reshape(1, -1)
                            q = row[model_cols].values.reshape(1, -1)
                            return cosine_similarity(p, q)[0][0]

                        df['Cosine_Similarity'] = df.apply(compute_cosine_similarity, axis=1)

                        def compute_hellinger_distance(row):
                            p = np.sqrt(row[annotator_cols].values)
                            q = np.sqrt(row[model_cols].values)
                            return np.linalg.norm(p - q) / np.sqrt(2)



                        def compute_mae(row):
                            p = row[annotator_cols].values
                            q = row[model_cols].values
                            return np.mean(np.abs(p - q))

                        df['MAE'] = df.apply(compute_mae, axis=1)

                        def compute_mse(row):
                            p = row[annotator_cols].values
                            q = row[model_cols].values
                            return np.mean((p - q) ** 2)

                        df['MSE'] = df.apply(compute_mse, axis=1)

                    #metrics = ['JSD', 'Correlation', 'KL_Divergence', 'Cosine_Similarity', 'MAE', 'MSE']
                    metrics = ['JSD', 'Correlation','MSE']
                    
                    # Create a new dataframe with only 3 rows (one for each seed)
                    summary_df = pd.DataFrame(columns= metrics, index = [0,1,2], dtype=np.float64)

                    for metric in metrics:
                        summary_df.loc[0,metric] = tabels[0][metric].mean()
                        summary_df.loc[1,metric] = tabels[1][metric].mean()
                        summary_df.loc[2,metric] = tabels[2][metric].mean()

                    # Calculate statistics (we care about mean and std) over the 3 seeds
                    summary_stats = summary_df[metrics].describe()
                    #print("summary_stats for", current_evaluation)
                    #print(summary_stats)
                    
                    """
                    make a struct like:
                    {
                        dataset: go_emotions,
                        technique: random,
                        model: bert,
                        table: {columns and rows}
                    }
                    """
                    print("summary_stats", summary_stats)   
                    #summary_stats = self.bold_extreme_values(summary_stats, extreme='max')
                    latex_table_helper_list.append({
                        "dataset": ds,  
                        "technique": tech,
                        "model": model,
                        "table": summary_stats
                    })
                    print("summary_stats", summary_stats)
                    
                    #LaTeX table
                    latex_table = summary_stats.to_latex(
                        index=False,
                        float_format="%.2f",
                        na_rep='-',
                        longtable=True,
                        caption="Evaluation Metrics for calculate_JSD_MSE_CORR",
                        label="tab:evaluation_metrics"
                    )
                    model_path_fix = model.replace("/", "-")
                    latex_path = os.path.join(self.latex_dir, f'calculate_JSD_MSE_CORR_{model_path_fix}_{ds}_{tech}_{idx + 1}.tex')
                    with open(latex_path, 'w') as f:
                        f.write(latex_table)
                    logger.info(f"LaTeX correlation table saved to {latex_path}")
                    #save MD
                    md_path = os.path.join(self.markdown_dir, f'calculate_JSD_MSE_CORR_{model_path_fix}_{ds}_{tech}_{idx + 1}.md')
                    with open(md_path, 'w') as f:
                        f.write(summary_stats.to_markdown())
                    logger.info(f"Markdown table saved to {md_path}")

        #print(latex_table_helper_list)
        l_table = self.generate_jsd_corr_mse_table(latex_table_helper_list)
        improved_latex_path = os.path.join(self.latex_dir, f'improved_calculate_JSD_MSE_CORR_combined.tex')
        with open(improved_latex_path, 'w') as f:
            f.write(l_table)
        logger.info(f"improved LaTeX table saved to {improved_latex_path}")



    def generate_jsd_corr_mse_table(self, data):
        from collections import OrderedDict

        aggregated = {}  # key: (dataset, technique) -> dict with aggregated values.
        counts = {}
        for entry in data:
            ds = entry['dataset']
            tech = entry['technique']
            tbl = entry['table'] 
            jsd_mean  = tbl.loc['mean', 'JSD'] # this is actually better than using the idx
            jsd_std   = tbl.loc['std',  'JSD']
            corr_mean = tbl.loc['mean', 'Correlation']
            corr_std  = tbl.loc['std',  'Correlation']
            mse_mean  = tbl.loc['mean', 'MSE']
            mse_std   = tbl.loc['std',  'MSE']
            key = (ds, tech)
            if key not in aggregated:
                aggregated[key] = {
                    'JSD_mean': 0, 'JSD_std': 0,
                    'Correlation_mean': 0, 'Correlation_std': 0,
                    'MSE_mean': 0, 'MSE_std': 0
                }
                counts[key] = 0
            aggregated[key]['JSD_mean'] += jsd_mean
            aggregated[key]['JSD_std']  += jsd_std
            aggregated[key]['Correlation_mean'] += corr_mean
            aggregated[key]['Correlation_std']  += corr_std
            aggregated[key]['MSE_mean']   += mse_mean
            aggregated[key]['MSE_std']    += mse_std
            counts[key] += 1

        for key in aggregated:
            aggregated[key]['JSD_mean'] /= counts[key]
            aggregated[key]['JSD_std']  /= counts[key]
            aggregated[key]['Correlation_mean'] /= counts[key]
            aggregated[key]['Correlation_std']  /= counts[key]
            aggregated[key]['MSE_mean']   /= counts[key]
            aggregated[key]['MSE_std']    /= counts[key]

        table_rows = []
        all_datasets = self.datasets #sorted({entry['dataset'] for entry in data})
        all_techniques = self.techniques # sorted({entry['technique'] for entry in data})

        for ds in all_datasets:
            for tech in all_techniques:
                if tech == None:
                    row = {
                        'Dataset': self.mapping_helper.get(ds, ds),
                        'Technique': 'Failed to load',
                        'Mean JSD': '',
                        'Mean Correlation': '',
                        'Mean MSE': ''
                    }
                    table_rows.append(row)
                else:
                    key = (ds, tech)
                    if key in aggregated:
                        vals = aggregated[key]
                        row = {
                            'Dataset': self.mapping_helper.get(ds, ds),
                            'Technique': self.mapping_helper.get(tech, tech),
                            'Mean JSD': f"{vals['JSD_mean']:.3f} $\\pm$ {vals['JSD_std']:.3f}",
                            'Mean Correlation': f"{vals['Correlation_mean']:.3f} $\\pm$ {vals['Correlation_std']:.3f}",
                            'Mean MSE': f"{vals['MSE_mean']:.4f} $\\pm$ {vals['MSE_std']:.4f}"
                        }
                        table_rows.append(row)
                    else:
                        pass

        # rows by dataset for multirow.
        grouped_rows = OrderedDict()
        for row in table_rows:
            ds = row['Dataset']
            if ds not in grouped_rows:
                grouped_rows[ds] = []
            grouped_rows[ds].append(row)


        latex_lines = [] # TODO do the other tables creation like this ... so we can skip the \n on each line
        latex_lines.append("\\begin{table*}[htbp]")
        latex_lines.append("    \\centering")
        latex_lines.append("    \\small")
        latex_lines.append("")
        latex_lines.append("    \\begin{tabular}{llccc}")
        latex_lines.append("    \\toprule")
        latex_lines.append("    \\textbf{Dataset} & \\textbf{Technique} & \\textbf{Mean JSD $\\downarrow$}  & \\textbf{Mean Correlation $\\uparrow$} & \\textbf{Mean MSE $\\downarrow$} \\\\")
        latex_lines.append("    \\midrule")
        
        dataset_keys = list(grouped_rows.keys())
        for i, ds in enumerate(dataset_keys):
            rows = grouped_rows[ds]
            n_rows = len(rows)
            for j, row in enumerate(rows):
                if row['Technique'] == 'Oracle':
                    latex_lines.append(" \\cmidrule{2-5}")
                if j == 0:
                    latex_lines.append(f"    \\multirow{{{n_rows}}}{{*}}{{{row['Dataset']}}} & {row['Technique']} & {row['Mean JSD']} & {row['Mean Correlation']} & {row['Mean MSE']} \\\\")
                else:
                    latex_lines.append(f"     & {row['Technique']} & {row['Mean JSD']} & {row['Mean Correlation']} & {row['Mean MSE']} \\\\")
            if i < len(dataset_keys) - 1:
                latex_lines.append("    \\midrule")
        latex_lines.append("    \\bottomrule")
        latex_lines.append("    \\end{tabular}")
        models_used = sorted({self.mapping_helper[entry['model']] for entry in data})
        models_str = ", ".join(models_used)
        latex_lines.append(f"        \\caption{{Aggregated Mean Results (averaged over {models_str})}}")
        latex_lines.append("    \\label{table:results_jsd_correlation_mse}")
        latex_lines.append("    \\vspace{0.2cm}")
        latex_lines.append("\\end{table*}")

        return "\n".join(latex_lines)


    def proof_of_concept_ambiguity_sample_detection(self, threshold_range_start = 60, threshold_range_end = 60, threshold_agreement_start = 60, threshold_agreement_end = 60,use_random_evaluation = False, text_scale = 1.0):
        summary_data = [] # TODO this shold be recreated on each model run, now it accumulates the tables ... 
        roc_data = {}
        
        if use_random_evaluation:
            print("Appending random evaluation")
            techniques = self.techniques + ['random'] # dont change the class variable
        else:
            print("Not appending random evaluation")
            techniques = self.techniques
        print("DEBUG class self.techniques:", self.techniques)
        
        print("Running proof of concept ambiguity detection")
        for model in self.models:
            for ds in self.datasets:
                for tech in techniques:
                    error_rates = []
                    percentages = []
                    accuracies = []
                    precisions = []
                    recalls = []
                    f1_scores = []
                    aucs = []
                    fprs = []
                    tprs = []
                    predictors = None

                    for i in range(1, 4):  # 1, 2, 3
                        current_evaluation = f"{model}_{ds}_{tech}_{i}"

                        if tech == 'random':
                            placeholder_name = f"{model}_{ds}_baseline_{i}"
                            random_threshold = True

                            workload = Workload.load(
                                workload_name=placeholder_name,
                                load_models=False,
                                quiet=True 
                            )
                            workload.model_name = 'random'
                            workload.name = current_evaluation
                        else:
                            random_threshold = False
                            workload = Workload.load(
                                workload_name=current_evaluation,
                                load_models=False,
                                quiet=True 
                            )

                        self.label_columns = self.load_labels(workload)
                        #invert entropy to work with the detection algorithm
                        workload.df['entropy_agreement'] = workload.df.apply(lambda row: self.compute_entropy_invert(row, self.label_columns), axis=1) 
                        
                        df = workload.df

                        best_threshold_combination, lowest_error_rate, results_df = workload.find_best_threshold_combination_based_on_error_rate(
                            agreement_column='entropy_agreement',
                            min_threshold=threshold_range_start,
                            max_threshold=threshold_range_end,
                            min_agreement_threshold=threshold_agreement_start,
                            max_agreement_threshold=threshold_agreement_end,
                            random_threshold=random_threshold,
                            columns=['entropy_mean_prediction', 'mean_variance', 'mean_jsd']
                        )
                        
                        error_rates.append(lowest_error_rate)
                        
                        if predictors is None:
                            predictors = workload.df[['entropy_mean_prediction', 'mean_variance', 'mean_jsd']]
                        else:
                            predictors = predictors + workload.df[['entropy_mean_prediction', 'mean_variance', 'mean_jsd']]

                        annotator_cols = [col for col in df.columns if col.endswith('_probability') and not col.endswith('_probability_model')]
                        model_cols = [col for col in df.columns if col.endswith('_probability_model')]
                        predicted_scores = df[model_cols].values
                        true_labels = df[annotator_cols].idxmax(axis=1).str.replace('_probability', '')

                        fpr, tpr, thresholds = roc_curve(workload.df['ambiguous'], workload.df['ambiguous_based_on_agreement'])
                        auc = roc_auc_score(workload.df['ambiguous'], workload.df['ambiguous_based_on_agreement'])
                        aucs.append(auc)
                        fprs.append(fpr)
                        tprs.append(tpr)

                        accuracy, precision, recall, f1 = workload.show_eval_metrics_acc_recall_prec_f1()
                        accuracies.append(accuracy)
                        precisions.append(precision)
                        recalls.append(recall)
                        f1_scores.append(f1)        

                    mean_error_rate = np.mean(error_rates)
                    std_error_rate = np.std(error_rates, ddof=1) 
                    mean_percentage = np.mean(percentages)
                    std_percentage = np.std(percentages, ddof=1) 
                    mean_accuracy = np.mean(accuracies)
                    std_accuracy = np.std(accuracies, ddof=1)
                    mean_precision = np.mean(precisions)
                    std_precision = np.std(precisions, ddof=1)
                    mean_recall = np.mean(recalls)
                    std_recall = np.std(recalls, ddof=1)
                    mean_f1 = np.mean(f1_scores)
                    std_f1 = np.std(f1_scores, ddof=1)
                    mean_auc = np.mean(aucs)
                    std_auc = np.std(aucs, ddof=1)
                    
                    mean_predictors = predictors / 3

                    summary_data.append({
                        'Model': model,
                        'Dataset': ds,
                        'Technique': tech,
                        'Mean Lowest Error Rate': mean_error_rate,
                        'STD Lowest Error Rate': std_error_rate,
                        'Mean Percentage': mean_percentage,
                        'STD Percentage': std_percentage,
                        'Mean Accuracy': mean_accuracy,
                        'STD Accuracy': std_accuracy,
                        'Mean Precision': mean_precision,
                        'STD Precision': std_precision,
                        'Mean Recall': mean_recall,
                        'STD Recall': std_recall,
                        'Mean F1 Score': mean_f1,
                        'STD F1 Score': std_f1,
                        'Mean AUC': mean_auc,
                        'STD AUC': std_auc
                    })
                    

                    error_rates_array = np.array(error_rates)
                    mean_error_rate_idx = np.argmin(np.abs(error_rates_array - mean_error_rate))
                    

                    print("fprs", fprs)
                    print("roc_auc", aucs)
                    roc_fpr = (fprs[0] + fprs[1] + fprs[2]) / 3
                    roc_tpr = (tprs[0] + tprs[1] + tprs[2]) / 3
                    roc_auc = (aucs[0] + aucs[1] + aucs[2]) / 3

                    roc_data[tech] = {
                        'fpr': roc_fpr,
                        'tpr': roc_tpr,
                        'auc': roc_auc
                    }
                    
                    y_true = workload.df['ambiguous_based_on_agreement'].astype(int)  
                    predictors_names = ['entropy_mean_prediction', 'mean_variance', 'mean_jsd']
                    #print(mean_predictors)
                    for predictor in predictors_names:
                        y_scores = mean_predictors[predictor]
                        fpr, tpr, thresholds = roc_curve(y_true, y_scores)
                        #roc_auc = auc(fpr, tpr)
                        plt.plot(fpr, tpr, label=f'{predictor} (AUC = {roc_auc:.2f})')
                    plt.xlim([0.0, 1.0])
                    plt.ylim([0.0, 1.05])
                    plt.xlabel('False Positive Rate')
                    plt.ylabel('True Positive Rate')
                    plt.title(f'ROC Curves for Ambiguity Detection for {current_evaluation}\n')
                    plt.legend(loc='lower right')
                    #plt.show()
                    model_path_fix = current_evaluation.replace("/", "-")

                summary_df = pd.DataFrame(summary_data)
                
                #print(summary_df)
                
                #print("=== Summary Table ===\n")
                #print(summary_df.to_string(index=False))
                latex_table = summary_df.to_latex(
                    index=False,
                    float_format="%.2f",
                    na_rep='-',
                    longtable=True,
                    caption="Evaluation Metrics for proof_of_concept_ambiguity_sample_detections",
                    label="tab:evaluation_metrics"
                )
                model_path_fix = current_evaluation.replace("/", "-")

                latex_path = os.path.join(self.latex_dir, f'proof_of_concept_{threshold_range_start}_{threshold_range_end}_{threshold_agreement_start}_{threshold_agreement_end}_ambiguity_sample_detections_{model_path_fix}.tex')
                with open(latex_path, 'w') as f:
                    f.write(latex_table)
                logger.info(f"LaTeX correlation table saved to {latex_path}")
                
                #save MD
                md_path = os.path.join(self.markdown_dir, f'proof_of_concept_{threshold_range_start}_{threshold_range_end}_{threshold_agreement_start}_{threshold_agreement_end}_ambiguity_sample_detections_{model_path_fix}.md')
                with open(md_path, 'w') as f:
                    f.write(summary_df.to_markdown())
                logger.info(f"Markdown table saved to {md_path}")

                plt.figure(figsize=(10, 8))
                for tech in techniques:
                    fpr = roc_data[tech]['fpr']
                    tpr = roc_data[tech]['tpr']
                    auc = roc_data[tech]['auc']
                    if tech == 'random':
                        plt.plot(fpr, tpr,  'k--', label=f"{tech.capitalize()} (AUC = {auc:.2f})")
                    else:
                        plt.plot(fpr, tpr, label=f"{tech.capitalize()} (AUC = {auc:.2f})")
    
                plt.xlabel('False Positive Rate', fontsize=14 * text_scale)
                plt.ylabel('True Positive Rate', fontsize=14 * text_scale)
                plt.title('ROC Curves for Mean Runs of Each Technique', fontsize=16 * text_scale)
                plt.legend(fontsize=12 * text_scale)
                plt.grid(True, ls="--", linewidth=0.5)
                plt.tight_layout()
                #plt.show()
                model_path_fix = current_evaluation.replace("/", "-")
                plt.savefig(os.path.join(self.plot_dir, f'proof_of_concept_{threshold_range_start}_{threshold_range_end}_{threshold_agreement_start}_{threshold_agreement_end}_ambiguity_sample_detections_{model_path_fix}.png'))
                plt.close()
                print(f"\nCombined ROC curves saved to '{self.plot_dir}'.")
        print("Finished proof of concept ambiguity sample detection")
        
        
        # average over models TODO consider putting this in a new method ... 
        if not summary_df.empty:
            agg_error = summary_df.groupby(['Technique', 'Dataset'], as_index=False).agg({
                'Mean Lowest Error Rate': ['mean', 'std']
            })

            agg_error.columns = ['Technique', 'Dataset', 'Mean Error Rate', 'STD Error Rate']
            error_pivot = agg_error.pivot(index='Technique', columns='Dataset', values=['Mean Error Rate', 'STD Error Rate'])

            #dataset_order = ['rt', 'go_emotions', 'hate_gap'] # TODO make this dynamic just parse the models ... 
            
            # TODO DEBUG use mapping helper function, tho they would override other mappings ... 
            display_names = { 
                'rt': "\\textbf{RT (\%)}",
                'go_emotions': "\\textbf{GoEmotions (\%)}",
                'hate_gap': "\\textbf{GAB Hate (\%)}"
            }
            technique_column_name = "\\textbf{Technique}"
            final_data = {}
            for tech in error_pivot.index:
                row = {}
                for ds in self.datasets:
                    try:
                        mean_val = error_pivot.loc[tech, ('Mean Error Rate', ds)]
                        std_val = error_pivot.loc[tech, ('STD Error Rate', ds)]
                        row[display_names[ds]] = f"{mean_val:.2f} $\\pm$ {std_val:.3f}"
                    except KeyError:
                        row[display_names[ds]] = ""
                final_data[tech] = row
            aggregated_error_df = pd.DataFrame.from_dict(final_data, orient='index')
            aggregated_error_df.index.name = technique_column_name
            aggregated_error_df.reset_index(inplace=True)
            
            # TODO DEBUG use mapping helper function tho they would override other mappings .... 
            technique_rename = {
                'baseline': "B (Baseline)",
                'mc': "MCD",
                'smoothing': "LS",
                'de': "DE",
                'random': "Random Chance",
                'ub': "Oracle",
                'Upper Bound': "Oracle"
            }
            aggregated_error_df[technique_column_name] = aggregated_error_df[technique_column_name].map(technique_rename).fillna(aggregated_error_df[technique_column_name])
            #aggregated_error_df = self.bold_extreme_values(aggregated_error_df, value_cols=["RT (%)", "GoEmotions (%)", "GAB Hate (%)"], extreme='min')
            #print(aggregated_error_df)
            aggregated_error_df = self.bold_extreme_values(aggregated_error_df, extreme='min')
            #print(aggregated_error_df)
            
            # TODO is there a way to add \centered and \small !?
            error_table_latex = aggregated_error_df.to_latex(
                index=False,
                escape=False,
                # TODO this is just the copied text, add a placeholder or even generate the results .... 
                caption=("Error Rates for Ambiguity Detection (Averaged over Models). "
                        "\\textbf{Note: REPLACE ME !} consistently achieves the lowest error rates, with the most pronounced improvement on the \\textbf{Note: REPLACE ME !} dataset."),
                label="table:aggregated_error_rate",
                column_format=f"l{'c'*len(self.datasets)}",  # dynamically creates "lccc" if len(self.datasets)==3
                bold_rows=False,
                float_format="{:.2f}".format # auto format to 2 decimal
            )
            
            error_latex_path = os.path.join(self.latex_dir, "aggregated_error_rate.tex")
            with open(error_latex_path, 'w') as f:
                f.write(error_table_latex)
            logger.info(f"Aggregated error rate LaTeX table saved to {error_latex_path}")


    # TODO write documentation for this function. Also use it on the other LateX tables ... 
    def bold_extreme_values(self, df, value_cols=None, extreme='min', bold_func=None):
        import re
        if value_cols is None:
            value_cols = df.columns.tolist()
            
        if bold_func is None:
            bold_func = lambda cell, val: f"\\textbf{{{cell}}}"
        
        def get_numeric(cell):
            try:
                if not isinstance(cell, str):
                    #print(f"´Cell is alead in float: {cell}")
                    return float(cell)
                #print(f"Cell is string: {cell}")
                if "$\\pm$" in cell:
                    num = float(cell.split(" $\\pm$ ")[0]) # TODO if contains a $\\pm$ we need to split it ...  
                else:
                    num = cell
                cleaned = re.sub(r"[^\d\.\-+]", "", num)
                #print(f"Cleaned: {num} -> {cleaned}")
                return float(cleaned)
            except Exception:
                try:
                    return float(cell)
                except Exception:
                    print(f"Failed to convert cell to float: {cell}")
                    return None
        #print(df)
        df_methods = df.loc[df['Technique'] != 'Oracle', :]
        
        for col in value_cols:
            numeric_vals = [get_numeric(cell) for cell in df_methods[col] if get_numeric(cell) is not None]
            if not numeric_vals:
                continue

            if extreme == 'min':
                extreme_val = min(numeric_vals)
            elif extreme == 'max':
                extreme_val = max(numeric_vals)
            else:
                raise ValueError("Parameter 'extreme' must be 'min' or 'max'.")
            # this is safe, as long as the func only wraps in \textbf{}
            df[col] = df[col].apply(
                lambda cell: bold_func(cell, extreme_val) 
                if (get_numeric(cell) is not None and np.isclose(get_numeric(cell), extreme_val, atol=1e-6))
                else cell
            )
        return df




    def proof_of_concept_ambiguity_sample_detection_latex_tabel(self, threshold_range_start = 60, threshold_range_end = 60, threshold_agreement_start = 60, threshold_agreement_end = 60):
        results_list = []

        for ds in self.datasets:
            for tech in techniques:
                for model in self.models:
                    error_rates = []
                    precisions = []
                    recalls = []
                    f1_scores = []
                    
                    for i in range(3):  
                        idx = (i % 3) + 1
                        current_evaluation = f"{model}_{ds}_{tech}_{idx}"
                        print("Running on", current_evaluation)
                        
                        if tech == 'random':
                            placeholder_name = f"{model}_{ds}_baseline_{idx}"
                            random_threshold = True

                            workload = Workload.load(
                                workload_name=placeholder_name,
                                load_models=False,
                                quiet=True 
                            )
                            workload.model_name = 'random'
                            workload.name = current_evaluation
                        else:
                            random_threshold = False
                            workload = Workload.load(
                                workload_name=current_evaluation,
                                load_models=False,
                                quiet=True 
                            )

                        workload.df['Model'] = model
                        workload.df['Run'] = idx

                        self.label_columns= self.load_labels(workload=workload)
                        workload.df['entropy_agreement'] = workload.df.apply(lambda row: self.compute_entropy_invert(row, self.label_columns), axis=1) 
                        
                        # Find the best threshold combination based on error rate

                        best_threshold_combination, lowest_error_rate, _ = workload.find_best_threshold_combination_based_on_error_rate(
                            agreement_column='entropy_agreement',
                            min_threshold=threshold_range_start,
                            max_threshold=threshold_range_end,
                            min_agreement_threshold=threshold_agreement_start,
                            max_agreement_threshold=threshold_agreement_end,
                            random_threshold=random_threshold,
                            columns=['entropy_mean_prediction', 'mean_variance', 'mean_jsd'],
                        )
   
                        accuracy, precision, recall, f1 = workload.show_eval_metrics_acc_recall_prec_f1()
                        print(f"Lowest Error Rate for {current_evaluation}: {lowest_error_rate}")

                        error_rates.append(lowest_error_rate)
                        precisions.append(precision)
                        recalls.append(recall)
                        f1_scores.append(f1)
                    
                    mean_error_rate = np.mean(error_rates)
                    std_error_rate = np.std(error_rates)
                    mean_precision = np.mean(precisions)
                    std_precision = np.std(precisions)
                    mean_recall = np.mean(recalls)
                    std_recall = np.std(recalls)
                    mean_f1 = np.mean(f1_scores)
                    std_f1 = np.std(f1_scores)
                    
                    if model in self.mapping_helper: # fix for models that are named like */* or unknown models.
                        model_name = self.mapping_helper[model]
                    else:
                        model_name = model
                    
                    results_list.append({
                        'Technique': self.mapping_helper[tech],
                        'Dataset': self.mapping_helper[ds],
                        'Model': model_name,
                        'Mean Error Rate': mean_error_rate,
                        'Std Error Rate': std_error_rate,
                        'Mean Precision': mean_precision,
                        'Std Precision': std_precision,
                        'Mean Recall': mean_recall,
                        'Std Recall': std_recall,
                        'Mean F1 Score': mean_f1,
                        'Std F1 Score': std_f1
                    })

        results_df = pd.DataFrame(results_list)
        latex_table = results_df.to_latex(
            index=False,
            float_format="%.2f",
            na_rep='-',
            longtable=True,
            caption="Evaluation Metrics for Models, Datasets, and Techniques",
            label="tab:evaluation_metrics"
        )


        latex_path = os.path.join(self.latex_dir, f'proof_of_concept_{threshold_range_start}_{threshold_range_end}_{threshold_agreement_start}_{threshold_agreement_end}_ambiguity_sample_detection_latex_tabel.tex')
        with open(latex_path, 'w') as f:
            f.write(latex_table)
        logger.info(f"LaTeX correlation table saved to {latex_path}")
        
        #save MD
        md_path = os.path.join(self.markdown_dir, f'proof_of_concept_{threshold_range_start}_{threshold_range_end}_{threshold_agreement_start}_{threshold_agreement_end}_ambiguity_sample_detection_latex_tabel.md')
        with open(md_path, 'w') as f:
            f.write(results_df.to_markdown())
        logger.info(f"Markdown table saved to {md_path}")        

        def format_metric(mean, std, is_percentage=False):
            if is_percentage:
                return f"{mean:.2f}% (± {std:.2f}%)"
            else:
                return f"{mean:.4f} (± {std:.4f})"

        results_df['Mean Error Rate'] = results_df.apply(
            lambda x: format_metric(x['Mean Error Rate'], x['Std Error Rate'], is_percentage=True), axis=1)
        results_df['Mean Precision'] = results_df.apply(
            lambda x: format_metric(x['Mean Precision'], x['Std Precision']), axis=1)
        results_df['Mean Recall'] = results_df.apply(
            lambda x: format_metric(x['Mean Recall'], x['Std Recall']), axis=1)
        results_df['Mean F1 Score'] = results_df.apply(
            lambda x: format_metric(x['Mean F1 Score'], x['Std F1 Score']), axis=1)

        results_df = results_df.drop(['Std Error Rate', 'Std Precision', 'Std Recall', 'Std F1 Score'], axis=1)

        results_df = results_df[['Technique', 'Dataset', 'Model', 'Mean Error Rate',
                                'Mean Precision', 'Mean Recall', 'Mean F1 Score']]


        #print(results_df)
        
        latex_table = results_df.to_latex(index=False, caption='Performance Metrics for Ambiguity Detection Techniques', label='table:case_study_results')
        #print(latex_table)
        latex_path = os.path.join(self.latex_dir, f'proof_of_concept_{threshold_range_start}_{threshold_range_end}_{threshold_agreement_start}_{threshold_agreement_end}_ambiguity_sample_detection_latex_tabel_formatted.tex')
        with open(latex_path, 'w') as f:
            f.write(latex_table)
            
        #save MD
        md_path = os.path.join(self.markdown_dir, f'proof_of_concept_{threshold_range_start}_{threshold_range_end}_{threshold_agreement_start}_{threshold_agreement_end}_ambiguity_sample_detection_latex_tabel_formatted.md')
        with open(md_path, 'w') as f:
            f.write(results_df.to_markdown())
        logger.info(f"Markdown table saved to {md_path}")    



    def prove_of_concept_ambiguity_sample_detection_combined_ROC(self, text_scale=1.0):


        summary_data = []

        roc_data = {tech: {ds: {'fpr': [], 'tpr': [], 'auc': []} for ds in self.datasets} for tech in techniques}

        for model in self.models:
            for ds in self.datasets:
                plt.figure(figsize=(20, 20))  
                #grid_size = 2 

                grid_size = math.ceil(math.sqrt(len(techniques)))
                subplot_idx = 1  
                
                for tech in techniques:
                    plt.subplot(grid_size, grid_size, subplot_idx)
                    plt.title(f"{self.mapping_helper[tech]} Technique", fontsize=16 * text_scale)
                    
                    predictors_accumulate = {
                        'entropy_mean_prediction': [],
                        'mean_variance': [],
                        'mean_jsd': []
                    }
                    
                    y_true = None
                    
                    for run in range(1, 4):  
                        current_evaluation = f"{model}_{ds}_{tech}_{run}"
                        
                        workload = Workload.load(
                            workload_name=current_evaluation,
                            load_models=False,
                            quiet=True  
                        )
                        
                        self.label_columns = self.load_labels(workload=workload)  
                        

                        workload.df['entropy_agreement'] = workload.df.apply(lambda row: self.compute_entropy_invert(row, self.label_columns), axis=1) 

                        workload.test_detection(
                            agreement_column="entropy_agreement",
                            default_threshold_percentile=60,
                            threshold_percentile_agreement=60,
                            columns=['entropy_mean_prediction', 'mean_variance', 'mean_jsd'],
                            quiet=True  
                        )
                        
                        df = workload.df
                        
                        if y_true is None:
                            y_true = df['ambiguous_based_on_agreement'].astype(int)
                        else:
                          
                            if not y_true.equals(df['ambiguous_based_on_agreement'].astype(int)):
                                print(f"Warning: y_true differs for {current_evaluation}")
                        
                        for predictor in ['entropy_mean_prediction', 'mean_variance', 'mean_jsd']:
                            predictors_accumulate[predictor].append(df[predictor].values)
                    
                    mean_predictors = {}
                    for predictor in ['entropy_mean_prediction', 'mean_variance', 'mean_jsd']:
                        stacked = np.stack(predictors_accumulate[predictor], axis=0)  # shape: (runs, samples)
                        mean_predictors[predictor] = np.mean(stacked, axis=0)  # shape: (samples,)
                    
                    for predictor in ['entropy_mean_prediction', 'mean_variance', 'mean_jsd']:
                        y_scores = mean_predictors[predictor]
                        fpr, tpr, thresholds = roc_curve(y_true, y_scores)
                        roc_auc_val = auc(fpr, tpr)
                        
                        # Skip plotting if AUC is 0.5 (random chance)
                        if roc_auc_val == 0.5:
                            continue
                        
                        roc_data[tech][ds]['fpr'].append(fpr)
                        roc_data[tech][ds]['tpr'].append(tpr)
                        roc_data[tech][ds]['auc'].append(roc_auc_val)
                        
                        plt.plot(fpr, tpr, label=f'{predictor} (AUC = {roc_auc_val:.2f})')
                        
                        summary_data.append({
                            'Technique': tech,
                            'Dataset': ds,
                            'Predictor': predictor,
                            'AUC': roc_auc_val
                        })
                    
                    plt.plot([0, 1], [0, 1], 'k--', label='Random Chance')
                    
                    plt.xlim([0.0, 1.0])
                    plt.ylim([0.0, 1.05])
                    plt.xlabel('False Positive Rate', fontsize=12 * text_scale)
                    plt.ylabel('True Positive Rate', fontsize=12 * text_scale)
                    plt.legend(loc='lower right', fontsize=8 * text_scale)
                    plt.grid(True, linestyle='--', linewidth=0.5 * text_scale)
                    
                    subplot_idx += 1
                
                plt.tight_layout(rect=[0, 0.03, 1, 0.95])
                plt.suptitle(f'ROC Curves for Techniques on {self.mapping_helper[ds]} Dataset', fontsize=20 * text_scale)
                
                model_path_fix = current_evaluation.replace("/", "-")
                plt.savefig(os.path.join(self.plot_dir, f'combined_ROC_{model_path_fix}.png'))
                print(f"\nROC curves for all techniques on {self.mapping_helper[ds]} saved to '{os.path.join(self.plot_dir, f'combined_ROC.png')}'.")
                
                #plt.show()
                plt.close()

            summary_df = pd.DataFrame(summary_data)

            #summary_csv_path = os.path.join(self.summary_dir, 'summary_table.csv')
            #summary_df.to_csv(summary_csv_path, index=False)
            #print(f"\nSummary table saved to '{summary_csv_path}'.\n")
            latex_table = summary_df.to_latex(
                index=False,
                float_format="%.2f",
                na_rep='-',
                longtable=True,
                caption="Evaluation Metrics for prove_of_concept_ambiguity_sample_detection_combined_ROC",
                label="tab:evaluation_metrics"
            )
            model_path_fix = current_evaluation.replace("/", "-")
            latex_path = os.path.join(self.latex_dir, f'prove_of_concept_ambiguity_sample_detection_combined_ROC_{model_path_fix}.tex')
            with open(latex_path, 'w') as f:
                f.write(latex_table)
            logger.info(f"LaTeX correlation table saved to {latex_path}")
            
            #save MD
            md_path = os.path.join(self.markdown_dir, f'prove_of_concept_ambiguity_sample_detection_combined_ROC_{model_path_fix}.md')
            with open(md_path, 'w') as f:
                f.write(summary_df.to_markdown())
            logger.info(f"Markdown table saved to {md_path}")    
            # Show the summary table
            #print("=== Summary Table ===\n")
            #print(summary_df.to_string(index=False))



    def plot_histrogram(self, column, workload_name, title, xlabel, ylabel, bins=10,show=True, save=False):
        # TODO put this into the plotting Class
        base_path = os.path.join(os.getcwd(), "saved_results", workload_name)
        df_path = os.path.join(base_path, "test.csv") # this could be a parameter
        if not os.path.exists(df_path):
            raise FileNotFoundError(f"The DataFrame file {df_path} does not exist.")
        df = pd.read_csv(df_path)
        # Debug:
        plt.figure(figsize = (8,3.3))
        print(f"DEBUG plot_histrogram for df: {workload_name} with column: {column} -> Min: {df['entropy_agreement'].min()}, Mean: {df['entropy_agreement'].mean()}, Max: {df['entropy_agreement'].max()}")
        plt.hist(df[column], bins=bins,  edgecolor='black')
        #plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        #plt.xlim((0,1.6))
        if save:
            save_path = os.path.join(os.getcwd(),'analysis_results', 'plots', f"histogram_{workload_name}_{column}.png")
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        if show:
            plt.show()
        plt.close()



if __name__ == "__main__":
    
    models = ["roberta", "bert", "xlnet"] # Models to use in the batch
    datasets = ['go_emotions', "rt", "hate_gap"] # Datasets to use in the batch
    #techniques = ["baseline", "mc", "smoothing", "de", "random"] # Techniques to use, tho only Baseline, MC Dropout, Label Smoothing and deep ensamble is currently implemented
    
    techniques = ['mc','baseline', "smoothing", "de", 'ub' ] # Techniques to use, tho only Baseline, MC Dropout, Label Smoothing and deep ensamble is currently implemented
    num_runs = 3  # Number of runs per technique, the models have to be trained and exists in the saved_results folder
    enable_plotting = True  # Set to False to disable plotting # TODO impl plotting and verbose output.



    # revert back to sorting, it will always be: baseine, DE, LS, MCD, Oracle
    techniques.sort(key=lambda t: 0 if t.lower() == 'baseline' else 1)

    evaluator = WorkloadEvaluator(models, datasets, techniques, num_runs, enable_plotting)
        
    evaluator.calculate_evaluation_metrics_for_base(print_latex=False)
    evaluator.ambiguity_human_vs_models_correlation()
    evaluator.scatter_plot_correlation_user_vs_models_entropy()
    evaluator.scatter_plot_correlation_user_vs_models_entropy_combined()
    evaluator.calculate_JSD_MSE_CORR()
    evaluator.proof_of_concept_ambiguity_sample_detection(threshold_range_start = 60, threshold_range_end = 60, threshold_agreement_start = 60, threshold_agreement_end = 60, text_scale=1.0, use_random_evaluation = True)
    evaluator.proof_of_concept_ambiguity_sample_detection_latex_tabel(threshold_range_start = 60, threshold_range_end = 60, threshold_agreement_start = 60, threshold_agreement_end = 60)

    #evaluator.proof_of_concept_ambiguity_sample_detection_latex_tabel() # this is called twice, so keep it out of the main function ...
    evaluator.prove_of_concept_ambiguity_sample_detection_combined_ROC(text_scale=1.5) # this could be merged with the other ROC plotting ... 
    
    
    # model and technique provide the same human annotator entroy, therefore just go over a random model and technique, also the runs
    # we load the test dataset, and plot the histogram of the entropy agreement
    # TODO Richard, put this method in the appropriate class .... 
    model = models[0]      
    technique = techniques[0] 
    for dataset in datasets:
        for run in range(1, 2):
            workload_name = f"{model}_{dataset}_{technique}_{run}"
            evaluator.plot_histrogram("entropy_agreement",
                workload_name,
                f"Histogram of {dataset}", 
                "Label Ambiguity Score", 
                "Number of samples", 
                bins=15,
                show=False,
                save=True)
