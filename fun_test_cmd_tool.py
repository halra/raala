import os
import sys
import logging

# Assume WorkloadEvaluator and other required classes and functions are imported properly
# from your_module import WorkloadEvaluator, train_and_evaluate_models

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s %(levelname)s:%(message)s')
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

def print_help():
    help_text = """
    === Help for the Script ===
    This script allows you to train models or analyze results interactively.
    
    Options:
    - Train: Train models with specified configurations.
    - Analyze: Analyze the results of trained models.

    For training, you will be prompted to enter:
    - Model names (e.g., 'bert-base,roberta-base')
    - Dataset name (e.g., 'hate_gap')
    - Techniques (e.g., 'baseline,mc,smoothing,de')
    - Seeds and epochs for each technique.
    - Number of samples for MC dropout runs.

    Note: For 'deep ensemble' technique, seeds and epochs should be arrays of arrays, as it requires multiple models.

    For analysis, you will be prompted to enter:
    - Models (e.g., 'bert-base,roberta-base')
    - Datasets (e.g., 'hate_gap,go_emotions,rt')
    - Techniques (e.g., 'baseline,mc,smoothing,de')
    - Number of runs per technique.
    - Whether to enable plotting.

    The script will display a summary before starting, which you can accept or decline.
    """
    print(help_text)

def main():
    print("=== Welcome to the Model Training and Analysis Tool ===")

    if len(sys.argv) > 1 and sys.argv[1] in ['-h', '--help']:
        print_help()
        return

    # Ask if the user wants to train or analyze
    while True:
        action = input("Do you want to (1) Train models, (2) Analyze results, or (3) Both? Enter 1, 2, or 3: ").strip()
        if action in ['1', '2', '3']:
            break
        else:
            print("Invalid input. Please enter 1, 2, or 3.")

    perform_training = action in ['1', '3']
    perform_analysis = action in ['2', '3']

    if perform_training:
        print("\n=== Model Training ===")
        # Ask user for model names
        model_names_input = input("Enter model names separated by commas (e.g., 'bert-base,roberta-base'): ")
        model_names = [name.strip() for name in model_names_input.split(',') if name.strip()]
        if not model_names:
            logger.error("No model names provided.")
            return

        # Ask for dataset name
        dataset_name = input("Enter dataset name: ").strip()
        if not dataset_name:
            logger.error("Dataset name cannot be empty.")
            return

        # Ask for techniques
        techniques_input = input("Enter techniques separated by commas (e.g., 'baseline,mc,smoothing,de'): ")
        techniques = [tech.strip() for tech in techniques_input.split(',') if tech.strip()]
        if not techniques:
            logger.error("No techniques provided.")
            return

        # For seeds_list and epochs_list, we need to match techniques
        seeds_list = []
        epochs_list = []
        print("\nEnter seeds and epochs for each technique.")

        for idx, tech in enumerate(techniques):
            print(f"\nTechnique: {tech}")
            if tech == 'de':
                print("Deep Ensemble requires arrays of seeds and epochs for each ensemble member.")
            seeds_input = input(f"  Enter seeds separated by commas (e.g., '42,13,815'):\n  For 'de', enter arrays separated by semicolons (e.g., '42,13;815,21'): ")
            try:
                if tech == 'de':
                    seeds_groups = [group.strip() for group in seeds_input.split(';') if group.strip()]
                    seeds = []
                    for group in seeds_groups:
                        seeds.append([int(s.strip()) for s in group.split(',') if s.strip()])
                    if not seeds:
                        logger.error(f"No seeds provided for technique '{tech}'.")
                        return
                    seeds_list.append(seeds)
                else:
                    seeds = [int(s.strip()) for s in seeds_input.split(',') if s.strip()]
                    if not seeds:
                        logger.error(f"No seeds provided for technique '{tech}'.")
                        return
                    seeds_list.append(seeds)
            except ValueError:
                logger.error(f"Invalid seed value provided for technique '{tech}'. Seeds must be integers.")
                return

            epochs_input = input(f"  Enter epochs separated by commas (e.g., '3,4,5'):\n  For 'de', enter arrays separated by semicolons (e.g., '3,4;5,6'): ")
            try:
                if tech == 'de':
                    epochs_groups = [group.strip() for group in epochs_input.split(';') if group.strip()]
                    epochs = []
                    for group in epochs_groups:
                        epochs.append([int(e.strip()) for e in group.split(',') if e.strip()])
                    if not epochs:
                        logger.error(f"No epochs provided for technique '{tech}'.")
                        return
                    epochs_list.append(epochs)
                else:
                    epochs = [int(e.strip()) for e in epochs_input.split(',') if e.strip()]
                    if not epochs:
                        logger.error(f"No epochs provided for technique '{tech}'.")
                        return
                    epochs_list.append(epochs)
            except ValueError:
                logger.error(f"Invalid epoch value provided for technique '{tech}'. Epochs must be integers.")
                return

        # Ask for num_samples (integer)
        num_samples_input = input("\nEnter number of samples for MC dropout runs (e.g., '100'): ")
        try:
            num_samples = int(num_samples_input)
        except ValueError:
            logger.error("Number of samples must be an integer.")
            return

        # Check inputs
        if len(techniques) != len(seeds_list) or len(techniques) != len(epochs_list):
            logger.error("Seeds, epochs, and techniques lists should be of equal length.")
            return

        # Show summary before starting
        print("\n=== Summary of Training Configuration ===")
        print(f"Models: {model_names}")
        print(f"Dataset: {dataset_name}")
        print(f"Techniques: {techniques}")
        print("Seeds List:")
        for tech, seeds in zip(techniques, seeds_list):
            print(f"  {tech}: {seeds}")
        print("Epochs List:")
        for tech, epochs in zip(techniques, epochs_list):
            print(f"  {tech}: {epochs}")
        print(f"Number of samples for MC dropout: {num_samples}")

        # Confirm to proceed
        proceed = input("\nDo you want to proceed with training? Enter 'yes' to proceed or 'no' to cancel: ").strip().lower()
        if proceed not in ['yes', 'y']:
            print("Training cancelled.")
            return

        # Now run the training
        train_and_evaluate_models(
            model_names=model_names,
            dataset_name=dataset_name,
            techniques=techniques,
            seeds_list=seeds_list,
            epochs_list=epochs_list,
            num_samples=num_samples
        )

    if perform_analysis:
        print("\n=== Analysis of Results ===")
        # Ask user for models
        models_input = input("Enter models separated by commas (e.g., 'bert-base,roberta-base'): ")
        models = [model.strip() for model in models_input.split(',') if model.strip()]
        if not models:
            logger.error("No models provided.")
            return

        # Ask for datasets
        datasets_input = input("Enter datasets separated by commas (e.g., 'hate_gap,go_emotions,rt'): ")
        datasets = [ds.strip() for ds in datasets_input.split(',') if ds.strip()]
        if not datasets:
            logger.error("No datasets provided.")
            return

        # Ask for techniques
        techniques_input = input("Enter techniques separated by commas (e.g., 'baseline,mc,smoothing,de'): ")
        techniques = [tech.strip() for tech in techniques_input.split(',') if tech.strip()]
        if not techniques:
            logger.error("No techniques provided.")
            return

        # Ask for num_runs
        num_runs_input = input("Enter number of runs per technique (e.g., '3'): ")
        try:
            num_runs = int(num_runs_input)
        except ValueError:
            logger.error("Number of runs must be an integer.")
            return

        # Ask for enable_plotting
        enable_plotting_input = input("Enable plotting? Enter 'yes' or 'no': ").strip().lower()
        if enable_plotting_input in ['yes', 'y', 'true', '1']:
            enable_plotting = True
        elif enable_plotting_input in ['no', 'n', 'false', '0']:
            enable_plotting = False
        else:
            logger.error("Invalid input for enable plotting. Please enter 'yes' or 'no'.")
            return

        # Show summary before starting
        print("\n=== Summary of Analysis Configuration ===")
        print(f"Models: {models}")
        print(f"Datasets: {datasets}")
        print(f"Techniques: {techniques}")
        print(f"Number of runs per technique: {num_runs}")
        print(f"Enable plotting: {enable_plotting}")

        # Confirm to proceed
        proceed = input("\nDo you want to proceed with analysis? Enter 'yes' to proceed or 'no' to cancel: ").strip().lower()
        if proceed not in ['yes', 'y']:
            print("Analysis cancelled.")
            return

        # Now run the evaluator
        evaluator = WorkloadEvaluator(models, datasets, techniques, num_runs, enable_plotting)
        evaluator.calculate_evaluation_metrics_for_base(print_latex=False)
        evaluator.ambiguity_human_vs_models_correlation()
        evaluator.scatter_plot_correlation_user_vs_models_entropy()
        evaluator.scatter_plot_correlation_user_vs_models_entropy_combined()
        evaluator.calculate_JSD_MSE_CORR()
        evaluator.proof_of_concept_ambiguity_sample_detection()
        evaluator.proof_of_concept_ambiguity_sample_detection_latex_tabel()
        evaluator.prove_of_concept_ambiguity_sample_detection_combined_ROC()

        # Provide some stats for the analysis_results folder
        print("\n=== Analysis Results Summary ===")
        analysis_results_dir = 'analysis_results'
        if os.path.exists(analysis_results_dir):
            for root, dirs, files in os.walk(analysis_results_dir):
                level = root.replace(analysis_results_dir, '').count(os.sep)
                indent = ' ' * 4 * (level)
                print(f"{indent}{os.path.basename(root)}/")
                subindent = ' ' * 4 * (level + 1)
                for f in files:
                    print(f"{subindent}{f}")
        else:
            print(f"No analysis results found in '{analysis_results_dir}'.")

    print("\nProcess completed.")

if __name__ == "__main__":
    main()
