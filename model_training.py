import os
import sys
import logging
from datetime import datetime
from typing import List

sys.path.append('./helper')
from domain.workload import Workload

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
file_handler = logging.FileHandler('./Logs/model_training.log')
console_handler.setLevel(logging.INFO)
file_handler.setLevel(logging.DEBUG)

formatter = logging.Formatter('%(asctime)s %(levelname)s:%(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

def train_and_evaluate_models( # TODO add dropout, smooting  and debug flags 
    model_names: List[str],
    dataset_name: str,
    techniques: List[str],
    seeds_list: List[List[int]],
    epochs_list: List[int],
    num_samples: int = 100
):
    for model_name in model_names:
        for idx, technique in  enumerate(techniques):
                seeds = seeds_list[idx]
                current_epochs = epochs_list[idx]
                suffix = (idx % 3)+ 1
                current_job_name = f"{model_name}_{dataset_name}_{technique}_{suffix}" # TODO make this idx better, make it more dynamic

                logger.info(f"Starting training for: {current_job_name}")

                workload = Workload.create(
                    name=current_job_name,
                    model_name=model_name,
                    seeds=seeds,
                    epochs=current_epochs
                )

                if technique == 'mc':
                    workload.train(dropout=0.5)
                    evaluation_type = 'mc_dropout'
                    workload.evaluate(
                        evaluation_type=evaluation_type,
                        debug=False,
                        num_samples=num_samples
                    )
                elif technique == 'smoothing':
                    workload.train(smoothing=0.3)
                    workload.evaluate(debug=False)
                else: # deep ensamble with one member is baseline
                    workload.train()
                    workload.evaluate(debug=False)

                workload.save()
                logger.info(f"Finished training for: {current_job_name}")

def main():
    #model_names = [
    #    'xlnet-base',
    #    'roberta-base',
    #    'bert-base'
    #]
    model_names = [
        'prajjwal1/bert-small'
    ]
    dataset_name = 'hate_gap'
    techniques = ['baseline','baseline','baseline', 'mc','mc','mc', 'smoothing','smoothing','smoothing', 'de', 'de','de']

    seeds_list = [
        [42], [13], [815],
        [42], [13], [815],
        [142, 113, 1815, 1142, 1113],
        [242, 213, 2815, 2142, 2113],
        [342, 313, 3815, 3142, 3113]
    ]
    epochs_list = [
        [3], [4], [5],
        [3], [4], [5],
        [3], [4], [5],
        [1, 2, 3, 4, 5],
        [1, 2, 3, 4, 5],
        [1, 2, 3, 4, 5]
    ]

    num_samples = 100  # mc dropout runs

    train_and_evaluate_models(
        model_names=model_names,
        dataset_name=dataset_name,
        techniques=techniques,
        seeds_list=seeds_list,
        epochs_list=epochs_list,
        num_samples=num_samples
    )

if __name__ == '__main__':
    main()