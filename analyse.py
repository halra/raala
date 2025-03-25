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
            'hate_gap': 'GAB Hate Corpus',
            'baseline': 'Baseline',
            'mc': 'MC Dropout',
            'smoothing': 'Label Smoothing',
            'de': 'Deep Ensemble',
            'prajjwal1/bert-small' :'prajjwal1-bert-small',
            'ub': 'Upper Bound',
            'entropy_agreement': 'Label Ambiguity Score',
            'entropy_mean_prediction': 'Model Uncertainty Score'
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

        
    
    def ambiguity_human_vs_models_correlation(self):
        latex_table_prepared =pd.DataFrame(columns=['Technique','Dataset','Model' ,'Mean Correlation','Std Dev %','Imp. over Baseline'])

        for model in models:
            for ds in datasets:
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

                    mean_std_corr_df['Mean Correlation'] = mean_std_corr_df['Mean Correlation'].round(4)
                    mean_std_corr_df['Std Correlation'] = mean_std_corr_df['Std Correlation'].round(4)
                    for idx in range(1, len(tabels) + 1):
                        diff_col_name = f'Diff_Corr_{idx}'
                        mean_std_corr_df[diff_col_name] = mean_std_corr_df[diff_col_name].round(4)

                    
                    base_value = 1
                    if tech == "baseline":
                        base_value = mean_std_corr_df['Mean Correlation'][1] # 1 is entropy
                    
                    if (mean_std_corr_df['Mean Correlation'][1] / base_value) < 1:
                        coeff = 1
                    else:
                        coeff = 1
                        
                    #print((mean_std_corr_df['Mean Correlation'][1] / base_value))
                    base_value = 1
                    
                    if model in self.mapping_helper:
                        model_name = self.mapping_helper[model]
                    else:
                        model_name = model
                    
                    #print(mean_std_corr_df['Mean Correlation'][1])
                    latex_table_prepared.loc[len(latex_table_prepared)] = [self.mapping_helper[tech],self.mapping_helper[ds], model_name, mean_std_corr_df['Mean Correlation'][1], mean_std_corr_df['Std Correlation'][1], str(  coeff *(- 100 + (mean_std_corr_df['Mean Correlation'][1] / base_value) * 100).round(1)) + "%"] 
                    #print(latex_table_prepared.loc[len(latex_table_prepared) - 1])
                    ######
                    latex_table = latex_table_prepared[['Technique','Dataset','Model','Mean Correlation','Std Dev %','Imp. over Baseline']].to_latex(index=False, 
                                        caption=f'Mean Correlation Coefficients between Uncertainty Metrics and Empirical Entropy (Average of 3 Runs)',
                                        label='table:roberta_rotten_tomatoes_correlations',
                                        column_format='lllcr',
                                        escape=True)
                    md_table_prepared = latex_table_prepared[['Technique','Dataset','Model','Mean Correlation','Std Dev %','Imp. over Baseline']]

                    #print(latex_table)
        #latex_table_prepared
        latex_table_prepared = pd.DataFrame(columns=['Technique','Dataset','Model','Mean Correlation','Std Dev %','Imp. over Baseline'])
        
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


    def scatter_plot_correlation_user_vs_models_entropy(self):
        for model in models:
            for ds in datasets:
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

        for ds in datasets:
            num_techs = len(techniques)
            num_cols = 2
            num_rows = math.ceil(num_techs / num_cols)
            fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 10))
            axes = axes.flatten()

            for idx_outer, tech in enumerate(techniques):
                ax = axes[idx_outer]
                all_data = []

                for model in models:
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
                    for idx_model, model in enumerate(models):
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
        for ds in datasets:
            num_techs = len(techniques)
            num_cols = 2
            num_rows = math.ceil(num_techs / num_cols)
            fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 10))
            axes = axes.flatten()
            
            # annotator[0.2,0.2,0.3,0.1,0.2] model[0.1,0.1,0.4,0.2,0.2]
            # JSD = 0.1 * log(0.1/0.2) + 0.2 * log(0.2/0.2) + 0.3 * log(0.3/0.4) + 0.1 * log(0.1/0.2) + 0.2 * log(0.2/0.2)
            # corr = np.corrcoef([0.2,0.2,0.3,0.1,0.2], [0.1,0.1,0.4,0.2,0.2])
            # now we sum up the jsd of each row and take the mean, this is the high std i guess
            for idx_outer, tech in enumerate(techniques):
                
                for model in models:
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


    def proof_of_concept_ambiguity_sample_detection(self, random_threshold=None):
        summary_data = [] # TODO this shold be recreated on each model run, now it accumulates the tables ... 
        roc_data = {}
        print("Running proof of concept ambiguity detection... with random_threshold", random_threshold)
        for model in models:
            for ds in datasets:
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
                        workload = Workload.load(
                            workload_name=current_evaluation,
                            load_models=False,
                            quiet=True 
                        )
                        
                        self.label_columns = self.load_labels(workload)
                        #invert entropy to work with the detection algorithm
                        workload.df['entropy_agreement'] = workload.df.apply(lambda row: self.compute_entropy_invert(row, self.label_columns), axis=1) 
                        
                        workload.test_detection(
                            agreement_column="entropy_agreement", 
                            default_threshold_percentile=60,
                            threshold_percentile_agreement=60,
                            columns=['entropy_mean_prediction', 'mean_variance', 'mean_jsd'],
                            quiet=True,
                            random_threshold=random_threshold
                        )
                        df = workload.df
                        filtered_df = df[(df['ambiguous'] == True) & (df['ambiguous_based_on_agreement'] == True)]
                        filtered_df_neg = df[(df['ambiguous'] == False) & (df['ambiguous_based_on_agreement'] == False)]
                        

                        percentage = ((len(filtered_df) + len(filtered_df_neg)) / len(df)) * 100
                        percentages.append(percentage)
                        
                        print(f"For Evaluation: {current_evaluation}")
                        print(f"Percentage of rows where ambiguous_based_on_agreement is True == True and False == False: {percentage:.2f}%\n")
                        
                        best_threshold_combination, lowest_error_rate, results_df = workload.find_best_threshold_combination_based_on_error_rate(
                            agreement_column='entropy_agreement',
                            min_threshold=60,
                            max_threshold=95,
                            min_agreement_threshold=60,
                            max_agreement_threshold=95,
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

                    median_error_rate = np.median(error_rates)
                    std_error_rate = np.std(error_rates, ddof=1) 
                    median_percentage = np.median(percentages)
                    std_percentage = np.std(percentages, ddof=1) 
                    median_accuracy = np.median(accuracies)
                    std_accuracy = np.std(accuracies, ddof=1)
                    median_precision = np.median(precisions)
                    std_precision = np.std(precisions, ddof=1)
                    median_recall = np.median(recalls)
                    std_recall = np.std(recalls, ddof=1)
                    median_f1 = np.median(f1_scores)
                    std_f1 = np.std(f1_scores, ddof=1)
                    median_auc = np.median(aucs)
                    std_auc = np.std(aucs, ddof=1)
                    
                    mean_predictors = predictors / 3

                    summary_data.append({
                        'Technique': tech,
                        'Median Lowest Error Rate': median_error_rate,
                        'STD Lowest Error Rate': std_error_rate,
                        'Median Percentage': median_percentage,
                        'STD Percentage': std_percentage,
                        'Median Accuracy': median_accuracy,
                        'STD Accuracy': std_accuracy,
                        'Median Precision': median_precision,
                        'STD Precision': std_precision,
                        'Median Recall': median_recall,
                        'STD Recall': std_recall,
                        'Median F1 Score': median_f1,
                        'STD F1 Score': std_f1,
                        'Median AUC': median_auc,
                        'STD AUC': std_auc
                    })
                    

                    error_rates_array = np.array(error_rates)
                    median_error_rate_idx = np.argmin(np.abs(error_rates_array - median_error_rate))
                    

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
                    plt.plot([0, 1], [0, 1], 'k--')  
                    plt.xlim([0.0, 1.0])
                    plt.ylim([0.0, 1.05])
                    plt.xlabel('False Positive Rate')
                    plt.ylabel('True Positive Rate')
                    plt.title(f'ROC Curves for Ambiguity Detection for {current_evaluation}\n')
                    plt.legend(loc='lower right')
                    #plt.show()
                    model_path_fix = current_evaluation.replace("/", "-")
                    if random_threshold is not None:
                        model_path_fix = f"{model_path_fix}_random_threshold_{random_threshold}"

                summary_df = pd.DataFrame(summary_data)
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
                if random_threshold is not None:
                    model_path_fix = f"{model_path_fix}_random_threshold_{random_threshold}"
                latex_path = os.path.join(self.latex_dir, f'proof_of_concept_ambiguity_sample_detections_{model_path_fix}.tex')
                with open(latex_path, 'w') as f:
                    f.write(latex_table)
                logger.info(f"LaTeX correlation table saved to {latex_path}")
                
                #save MD
                md_path = os.path.join(self.markdown_dir, f'proof_of_concept_ambiguity_sample_detections_{model_path_fix}.md')
                with open(md_path, 'w') as f:
                    f.write(summary_df.to_markdown())
                logger.info(f"Markdown table saved to {md_path}")

                plt.figure(figsize=(10, 8))
                for tech in techniques:
                    fpr = roc_data[tech]['fpr']
                    tpr = roc_data[tech]['tpr']
                    auc = roc_data[tech]['auc']
                    plt.plot(fpr, tpr, label=f"{tech.capitalize()} (AUC = {auc:.2f})")
                    
                plt.plot([0, 1], [0, 1], 'k--', label='Random Chance')

                plt.xlabel('False Positive Rate', fontsize=14)
                plt.ylabel('True Positive Rate', fontsize=14)
                plt.title('ROC Curves for Median Runs of Each Technique', fontsize=16)
                plt.legend(fontsize=12)
                plt.grid(True, ls="--", linewidth=0.5)
                plt.tight_layout()
                #plt.show()
                model_path_fix = current_evaluation.replace("/", "-")
                if random_threshold is not None:
                    model_path_fix = f"{model_path_fix}_random_threshold_{random_threshold}"
                plt.savefig(os.path.join(self.plot_dir, f'proof_of_concept_ambiguity_sample_detections_{model_path_fix}.png'))
                plt.close()
                print(f"\nCombined ROC curves saved to '{self.plot_dir}'.")


    def proof_of_concept_ambiguity_sample_detection_latex_tabel(self):
        results_list = []

        for ds in datasets:
            for tech in techniques:
                for model in models:
                    error_rates = []
                    precisions = []
                    recalls = []
                    f1_scores = []
                    
                    for i in range(3):  
                        idx = (i % 3) + 1
                        current_evaluation = f"{model}_{ds}_{tech}_{idx}"
                        print("Running on", current_evaluation)
                        
                        workload = Workload.load(workload_name=current_evaluation, load_models=False, quiet=True)
                        workload.df['Model'] = model
                        workload.df['Run'] = idx
                        self.label_columns= self.load_labels(workload=workload)
                        workload.df['entropy_agreement'] = workload.df.apply(lambda row: self.compute_entropy_invert(row, self.label_columns), axis=1) 
                        
                        #detection test
                        workload.test_detection(
                            agreement_column="entropy_agreement",
                            default_threshold_percentile=60,
                            threshold_percentile_agreement=60,
                            columns=['entropy_mean_prediction', 'mean_variance', 'mean_jsd'],
                            quiet=True  
                        )
                        
                        # Find the best threshold combination based on error rate
                        best_threshold_combination, lowest_error_rate, _ = workload.find_best_threshold_combination_based_on_error_rate(
                            agreement_column='entropy_agreement',
                            min_threshold=60,
                            max_threshold=60,
                            min_agreement_threshold=60,
                            max_agreement_threshold=60,
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


        latex_path = os.path.join(self.latex_dir, 'proof_of_concept_ambiguity_sample_detection_latex_tabel.tex')
        with open(latex_path, 'w') as f:
            f.write(latex_table)
        logger.info(f"LaTeX correlation table saved to {latex_path}")
        
        #save MD
        md_path = os.path.join(self.markdown_dir, 'proof_of_concept_ambiguity_sample_detection_latex_tabel.md')
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

 


        latex_table = results_df.to_latex(index=False, caption='Performance Metrics for Ambiguity Detection Techniques', label='table:case_study_results')
        #print(latex_table)
        latex_path = os.path.join(self.latex_dir, 'proof_of_concept_ambiguity_sample_detection_latex_tabel_formatted.tex')
        with open(latex_path, 'w') as f:
            f.write(latex_table)
            
        #save MD
        md_path = os.path.join(self.markdown_dir, 'proof_of_concept_ambiguity_sample_detection_latex_tabel_formatted.md')
        with open(md_path, 'w') as f:
            f.write(results_df.to_markdown())
        logger.info(f"Markdown table saved to {md_path}")    



    def prove_of_concept_ambiguity_sample_detection_combined_ROC(self):


        summary_data = []

        roc_data = {tech: {ds: {'fpr': [], 'tpr': [], 'auc': []} for ds in datasets} for tech in techniques}

        for model in models:
            for ds in datasets:
                plt.figure(figsize=(20, 20))  
                grid_size = 2  
                subplot_idx = 1  
                
                for tech in techniques:
                    plt.subplot(grid_size, grid_size, subplot_idx)
                    plt.title(f"{self.mapping_helper[tech]} Technique", fontsize=16)
                    
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
                    plt.xlabel('False Positive Rate', fontsize=12)
                    plt.ylabel('True Positive Rate', fontsize=12)
                    plt.legend(loc='lower right', fontsize=8)
                    plt.grid(True, linestyle='--', linewidth=0.5)
                    
                    subplot_idx += 1
                
                plt.tight_layout(rect=[0, 0.03, 1, 0.95])
                plt.suptitle(f'ROC Curves for Techniques on {self.mapping_helper[ds]} Dataset', fontsize=20)
                
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



if __name__ == "__main__":
    
    #models = ['bert', 'roberta', 'xlnet'] # Models to use in the batch
    #datasets = ['hate_gap', 'go_emotions', 'rt'] # Datasets to use in the batch
    #techniques = ["baseline", "mc", "smoothing", "de"] # Techniques to use, tho only Baseline, MC Dropout, Label Smoothing and deep ensamble is currently implemented
    
    models = ['google-bert/bert-base-uncased'] # Models to use in the batch
    datasets = ['go_emotions'] # Datasets to use in the batch
    techniques = ["ub"] # Techniques to use, tho only Baseline, MC Dropout, Label Smoothing and deep ensamble is currently implemented
    num_runs = 3  # Number of runs per technique, the models have to be trained and exists in the saved_results folder
    enable_plotting = True  # Set to False to disable plotting # TODO impl plotting and verbose output.

    evaluator = WorkloadEvaluator(models, datasets, techniques, num_runs, enable_plotting)
    evaluator.calculate_evaluation_metrics_for_base(print_latex=False)
    evaluator.ambiguity_human_vs_models_correlation()
    evaluator.scatter_plot_correlation_user_vs_models_entropy()
    evaluator.scatter_plot_correlation_user_vs_models_entropy_combined()
    evaluator.calculate_JSD_MSE_CORR()
    evaluator.proof_of_concept_ambiguity_sample_detection()
    evaluator.proof_of_concept_ambiguity_sample_detection(random_threshold=0.5)
    evaluator.proof_of_concept_ambiguity_sample_detection_latex_tabel()
    evaluator.prove_of_concept_ambiguity_sample_detection_combined_ROC() # this could be merged with the other ROC plotting ... 

    