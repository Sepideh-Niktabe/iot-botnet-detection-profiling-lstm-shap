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

## SHAP Bar Plot
### SHAP profiling samples using bar plot individually 

![shap_bar_plot_multiclass_Gagfyt_original_96](https://github.com/user-attachments/assets/616363a8-3c78-496a-8335-4c9b0ad77d29)
![shap_bar_plot_multiclass_IRCBot_original_96](https://github.com/user-attachments/assets/e5c43a53-d02b-42c5-b0ca-d8fa308e998a)
![shap_bar_plot_multiclass_Kenjiro_original_96](https://github.com/user-attachments/assets/ce883848-3244-467c-9e12-4bfda53e2f6f)
![shap_bar_plot_multiclass_Linux,Mirai_original_96](https://github.com/user-attachments/assets/169b91bf-0554-4053-8e3f-0add4a586e50)
![shap_bar_plot_multiclass_Mirai_original_96](https://github.com/user-attachments/assets/5b487f8c-4f96-477d-94ac-ab629860e826)
![shap_bar_plot_multiclass_Okiru_original_96](https://github.com/user-attachments/assets/5a3dfe75-53ab-4c5d-919a-24835adde9e0)



### SHAP profiling using bar plot collectivelly
![shap_summary_plot_original_bar_single_96](https://github.com/user-attachments/assets/3420c0be-14cd-4767-a749-b53a69bd604e)


## SHAP Violin Plot
### SHAP profiling using violin plot individually 
![shap_violin_multiclass_stick_IRCBot_96](https://github.com/user-attachments/assets/a1029e39-4318-4112-b1f9-f1c9eacddf0d)
![shap_violin_multiclass_stick_Gagfyt_96](https://github.com/user-attachments/assets/1a2fd155-d0a0-4446-a262-42985e2a2b13)
![shap_violin_multiclass_stick_Kenjiro_96](https://github.com/user-attachments/assets/005d3455-3c71-4116-ac6c-87ffe0b944af)
![shap_violin_multiclass_stick_Linux,Mirai_96](https://github.com/user-attachments/assets/4ec3d086-7343-4e77-a2ee-3dde638b2213)
![shap_violin_multiclass_stick_Okiru_96](https://github.com/user-attachments/assets/c298b761-2891-44a7-9414-adeafaf2a220)
![shap_violin_multiclass_stick_Mirai_96](https://github.com/user-attachments/assets/7fe24378-eb61-4672-9d03-d6189f539986)

### SHAP profiling using violin plot collectivelly
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
 
