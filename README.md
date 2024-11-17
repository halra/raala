# Uncertainty Measurement for Detecting Ambiguous Samples in Datasets

This repository contains code used to explore how uncertainty measurements can be utilized to identify and detect ambiguous samples in datasets. As this is an ongoing research project, the code may undergo significant changes in the future.

## Table of Contents

- [Overview](#overview)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
- [Usage](#usage)
  - [Preparing Datasets](#preparing-datasets)
  - [Training Models](#training-models)
  - [Analyzing Results](#analyzing-results)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)
- [Contact](#contact)

## Overview

This project investigates the application of uncertainty measurements in machine learning models to detect ambiguous samples within datasets. By leveraging techniques like entropy measurement and model uncertainty, we aim to improve the quality of datasets and the robustness of models trained on them.

## Getting Started

### Prerequisites

- Python 3.6 or higher
- Required Python libraries:
  - [NumPy](https://numpy.org/)
  - [Pandas](https://pandas.pydata.org/)
  - [SciPy](https://www.scipy.org/)
  - [Matplotlib](https://matplotlib.org/)
  - [Seaborn](https://seaborn.pydata.org/)
  - [Scikit-learn](https://scikit-learn.org/stable/)
  - [Flair](https://github.com/flairNLP/flair)
- Access to the [Hugging Face Model Hub](https://huggingface.co/) for pre-trained models

### Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/halra/raala.git
   cd raala
   ```

2. **Create a virtual environment (optional):**

   ```bash
   python -m venv venv
   source venv/bin/activate
   ```

3. **Install the required packages:**

   # TODO provide requirements.txt
   ```bash
   pip install -r requirements.txt
   ```
## Usage

### Preparing Datasets

*To be documented.*

Any dataset can be used with this code if it is prepared correctly. Please refer to the `prepare_*` Python scripts for examples of how to format and prepare your datasets. Detailed documentation on dataset preparation is forthcoming.

### Training Models

1. **Select Models and Datasets:**

   Open the `model_training.py` script and specify the models and datasets you wish to use. Models can be selected from the [Hugging Face Model Hub](https://huggingface.co/models). For example:

   ```python
   model_names = [
       'prajjwal1/bert-small'  # Replace with desired model(s)
   ]
   dataset_name = 'hate_gab'  # Replace with your dataset name
   techniques = ['baseline', 'mc', 'smoothing', 'de'] # currentrly the only implementet techniques
   ```

2. **Set Training Parameters:**

   Each array correspondents to a technique.
   Define the seeds and epochs for training:

   ```python
   seeds_list = [
       [42], [13], [815], # 'baseline', 'mc', 'smoothing'
       [142, 113, 1815, 1142, 1113] # de
   ]
   epochs_list = [
       [3], [4], [5], # 'baseline', 'mc', 'smoothing'
       [1, 2, 3, 4, 5] # de
   ]
   ```

3. **Run the Training Script:**

   Execute the training script:

   ```bash
   python model_training.py
   ```

   The script will train the models and save them in the `models` directory. Evaluation results will be automatically generated and saved in the `saved_results` directory.

### Analyzing Results

1. **Run the Analysis Script:**

   After training, run the analysis script to evaluate the results:

   ```bash
   python analyze.py
   ```

2. **View Results:**

   The analysis script will generate plots, LaTeX tables, and CSV files (if enabled) and save them in the `analysis_results` directory.

## Results

All results, including models, evaluation metrics, plots, and tables, are saved in their respective directories:

- **Models:** `models/`
- **Evaluation Results:** `saved_results/`
- **Analysis Results:** `analysis_results/`

*Note: As this is a proof of concept, the code and results are provided as-is.*

## Contributing

Contributions are welcome! If you have suggestions for improvements or encounter any issues, please feel free to contact me.

## License

# TODO but will be free to use 

## Acknowledgments

This project makes extensive use of the following libraries and frameworks:

- [NumPy](https://numpy.org/)
- [Pandas](https://pandas.pydata.org/)
- [SciPy](https://www.scipy.org/)
- [Matplotlib](https://matplotlib.org/)
- [Seaborn](https://seaborn.pydata.org/)
- [Scikit-learn](https://scikit-learn.org/stable/)
- [Flair](https://github.com/flairNLP/flair)
- [Hugging Face Transformers](https://huggingface.co/transformers/)

We would like to acknowledge the developers and contributors of these open-source projects.

---

*Please note that all datasets used are for research purposes only and are not utilized in any commercial capacity. This project aims solely to contribute to the academic community and enhance understanding in the field of machine learning and data analysis.*