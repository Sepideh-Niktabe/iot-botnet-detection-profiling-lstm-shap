# IoT network Malicious Behaviour Profiling Based on Explainable AI Using LSTM and SHAP

## Overview

This project implements an optimized IoT bot detection and identification model using Explainable AI (XAI) techniques, specifically SHAP (SHapley Additive exPlanations). The model combines XGBoost with a novel feature selection method and an LSTM neural network optimized via Bayesian Optimization. It achieves exceptional performance with 0.9999 accuracy, precision, recall, and F1-score on the augmented BCCC-Aposemat-Bot-IoT-24 dataset, outperforming established models. The model effectively handles sequential data and imbalanced datasets while providing explainable insights.

The evaluation and deployment of the model are streamlined using AWS SageMaker and S3 storage.

## Key Features

- **Explainable AI (XAI):** Utilizes SHAP for explainability, offering insights into model decisions.
- **High Accuracy:** Achieves near-perfect accuracy, precision, recall, and F1-score on the test dataset.
- **Advanced Optimization:** Uses Bayesian Optimization to fine-tune LSTM hyperparameters.
- **Cloud Integration:** Employs AWS SageMaker for model building, training, and deployment.
- **Data Handling:** Excels in managing sequential and imbalanced IoT data.

### SHAP Bar Plot
## SHAP profiling samples using bar plot individually 
![shap_bar_plot_multiclass_Linux,Mirai_original_96](https://github.com/user-attachments/assets/e44a35fc-14ad-4901-806b-214394866a9a)
## SHAP profiling using bar plot collectivelly
![shap_summary_plot_original_bar_single_96](https://github.com/user-attachments/assets/3420c0be-14cd-4767-a749-b53a69bd604e)


### SHAP Violin Plot
## SHAP profiling samples using violin plot individually 
![shap_violin_multiclass_stick_Linux,Mirai_96](https://github.com/user-attachments/assets/e34e1184-8f05-4e8a-ba7c-42ba086e4a28)

## SHAP profiling using violin plot collectivelly
![shap_violin_horizontal_seaborn_96](https://github.com/user-attachments/assets/286bea35-4092-4c70-9aa9-ff56dbc553ee)


## Repository Structure

- `LSTM_hyper_tuning.ipynb`: Notebook for hyperparameter optimization using Bayesian Optimization.
- `lstm_with_tuning.py`: Script for LSTM model training with optimized hyperparameters.
- `load_run_visualization.ipynb`: Notebook for loading the trained model, running it, and visualizing results using SHAP with bar and violin plots.
- `load_run_visualization.py`: Script for training the optimized LSTM model and applying SHAP for explainable insights.

## Installation and Setup

1. **Clone the repository:**

   ```bash
   git clone https://github.com/yourusername/IoT-Bot-Detection.git
 
