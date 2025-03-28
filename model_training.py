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
    
    if len(techniques) != len(seeds_list) or len(techniques) != len(epochs_list):
        logger.error("Seeds, epochs and techniques lists should be of equal length")
        return
    for model_name in model_names:
        for idx, technique in  enumerate(techniques):
                seeds = seeds_list[idx]
                current_epochs = epochs_list[idx]
                suffix = (idx % 3)+ 1 # TODO the modul has to be dynamic
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
                elif technique == 'ub':
                    workload.train(training_type='ub')
                    workload.evaluate(debug=False, evaluation_type='ub') 
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
        # 'google-bert/bert-base-uncased', # done
        'FacebookAI/roberta-base',
        'xlnet/xlnet-base-cased'
    ]
    dataset_name = 'go_emotions' 
    
    # TODO consider making a struct for the train_and_evaluate_models function
    # TODO check that len of techniques is equal to len of seeds and epochs
    # TODO add a function to train with different hyperparameters and consider thresholding them
    techniques = ['ub', 'ub','ub']

    seeds_list = [
        [42], [13], [815], #baseline
    ]
    epochs_list = [
        [10], [10], [10], #baseline
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
