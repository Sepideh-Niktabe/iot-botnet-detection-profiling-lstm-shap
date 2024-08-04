# IoT Bot Detection and Identification Model

## Overview

This project implements an optimized IoT bot detection and identification model using Explainable AI (XAI) techniques, specifically SHAP (SHapley Additive exPlanations). The model combines XGBoost with a novel feature selection method and an LSTM neural network optimized via Bayesian Optimization. It achieves exceptional performance with 0.9999 accuracy, precision, recall, and F1-score on the augmented BCCC-Aposemat-Bot-IoT-24 dataset, outperforming established models. The model effectively handles sequential data and imbalanced datasets while providing explainable insights.

The evaluation and deployment of the model are streamlined using AWS SageMaker and S3 storage.

## Key Features

- **Explainable AI (XAI):** Utilizes SHAP for explainability, offering insights into model decisions.
- **High Accuracy:** Achieves near-perfect accuracy, precision, recall, and F1-score on the test dataset.
- **Advanced Optimization:** Uses Bayesian Optimization to fine-tune LSTM hyperparameters.
- **Cloud Integration:** Employs AWS SageMaker for model building, training, and deployment.
- **Data Handling:** Excels in managing sequential and imbalanced IoT data.

## Repository Structure

- `LSTM_hyper_tuning.ipynb`: Notebook for hyperparameter optimization using Bayesian Optimization.
- `lstm_with_tuning.py`: Script for LSTM model training with optimized hyperparameters.
- `load_run_visualization.ipynb`: Notebook for loading the trained model, running it, and visualizing results using SHAP with bar and violin plots.
- `load_run_visualization.py`: Script for training the optimized LSTM model and applying SHAP for explainable insights.

## Installation and Setup

1. **Clone the repository:**

   ```bash
   git clone https://github.com/yourusername/IoT-Bot-Detection.git
 
