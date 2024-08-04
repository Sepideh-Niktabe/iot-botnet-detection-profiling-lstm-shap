import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
import argparse
import os
import io  # Added import
import boto3
from sagemaker.experiments.run import Run, load_run
from sagemaker.session import Session

from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from datetime import datetime
from io import StringIO

import botocore
import json
from tensorflow.keras.models import Model
import shap




# Use the same experiment name as defined in the training script
experiment_name = "SHAP-main-model-performance-experiments-96" 

# Set the AWS region
aws_region = os.environ.get('AWS_REGION', 'us-east-2') 
boto_session= boto3.setup_default_session(region_name=aws_region)
sagemaker_session = Session(boto_session)
s3 = boto3.client("s3")

columns = ['mode_payload_bytes_delta_len', 'mode_fwd_packets_delta_len',
               'mean_bwd_payload_bytes_delta_len', 'cov_bwd_header_bytes_delta_len',
               'mode_packets_delta_len', 'cov_bwd_packets_delta_len', 'active_skewness',
               'median_bwd_packets_delta_len', 'bwd_payload_bytes_skewness',
               'mean_bwd_packets_delta_len', 'mode_bwd_payload_bytes_delta_len',
               'fwd_total_header_bytes', 'median_fwd_payload_bytes_delta_len',
               'skewness_bwd_header_bytes_delta_len', 'active_min',
               'mode_fwd_payload_bytes_delta_len', 'Sample Classification']



columns_features = ['mode_payload_bytes_delta_len', 'mode_fwd_packets_delta_len',
               'mean_bwd_payload_bytes_delta_len', 'cov_bwd_header_bytes_delta_len',
               'mode_packets_delta_len', 'cov_bwd_packets_delta_len', 'active_skewness',
               'median_bwd_packets_delta_len', 'bwd_payload_bytes_skewness',
               'mean_bwd_packets_delta_len', 'mode_bwd_payload_bytes_delta_len',
               'fwd_total_header_bytes', 'median_fwd_payload_bytes_delta_len',
               'skewness_bwd_header_bytes_delta_len', 'active_min',
               'mode_fwd_payload_bytes_delta_len']



# violin_features_orders=['mode_payload_bytes_delta_len', 'mean_bwd_payload_bytes_delta_len', 'bwd_payload_bytes_skewness','mode_bwd_payload_bytes_delta_len', 'mode_fwd_payload_bytes_delta_len','median_fwd_payload_bytes_delta_len','cov_bwd_header_bytes_delta_len','skewness_bwd_header_bytes_delta_len','fwd_total_header_bytes','mode_fwd_packets_delta_len',
# 'median_bwd_packets_delta_len','mean_bwd_packets_delta_len','mode_packets_delta_len','cov_bwd_packets_delta_len','active_skewness','active_min']


#Category of features First type: 


#Payload based: 'mode_payload_bytes_delta_len', 'mean_bwd_payload_bytes_delta_len', 'bwd_payload_bytes_skewness','mode_bwd_payload_bytes_delta_len', 'mode_fwd_payload_bytes_delta_len','median_fwd_payload_bytes_delta_len'
#Packet based: 'mode_fwd_packets_delta_len' ,'mode_packets_delta_len', 'cov_bwd_packets_delta_len','median_bwd_packets_delta_len','mean_bwd_packets_delta_len',
#Header based:'cov_bwd_header_bytes_delta_len','skewness_bwd_header_bytes_delta_len','fwd_total_header_bytes'
#Time based: 'active_skewness', 'active_min'


#Category of features Second type: 

#Length based : 'mode_payload_bytes_delta_len', 'mode_fwd_packets_delta_len','mean_bwd_payload_bytes_delta_len', 'cov_bwd_header_bytes_delta_len','mode_packets_delta_len', 'cov_bwd_packets_delta_len', 'median_bwd_packets_delta_len', 'bwd_payload_bytes_skewness','mean_bwd_packets_delta_len', 'mode_bwd_payload_bytes_delta_len', 'fwd_total_header_bytes', 'median_fwd_payload_bytes_delta_len','skewness_bwd_header_bytes_delta_len',  'mode_fwd_payload_bytes_delta_len'

#Time based : 'active_min', 'active_skewness'





def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--units', type=int, default=50)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_MODEL_DIR'))
    parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAIN'))
    parser.add_argument('--label_column', type=str, default='Sample Classification')
    # parser.add_argument('--run_name', type=str, required=True)
    # parser.add_argument('--trial_name', type=str, required=True)  # Accept trial name as an argument
    ##########
    parser.add_argument('--optimizer', type=str, default='Adam')
    parser.add_argument('--activation', type=str, default='tanh')
    parser.add_argument('--enable_shap', type=bool, default=False, help="Enable SHAP analysis (requires compatible versions)")
    return parser.parse_args()




class ExperimentCallback(tf.keras.callbacks.Callback):
    def __init__(self, run,model, x_test, y_test):
        self.run = run
        self.model = model
        self.x_test = x_test
        self.y_test = y_test
      

    def on_epoch_end(self, epoch, logs=None):
        keys = list(logs.keys())
        for key in keys:
            self.run.log_metric(name=key, value=logs[key], step=epoch)
            print(f"{key}={logs[key]}")

def load_data(label_column):
    # Specify bucket and file_key
    bucket = 'multiclass-balanced-two-million'
    file_key = 'merged-multiclass-balanced-two-million.csv'
    # file_key ='LSTM_random_multiclass_alllabels.csv'
    bucket_name = 'multiclass-balanced-two-million'
    folder_name = "SHAP-decrease-main-model-performance-experiments-96" 

    # Fetch data from S3
    s3_client = boto3.client('s3')
    csv_obj = s3_client.get_object(Bucket=bucket, Key=file_key)
    body = csv_obj['Body']
    csv_string = body.read().decode('utf-8')
    data = pd.read_csv(io.StringIO(csv_string), usecols=columns)

    # Prepare features and labels
    x = data.drop(label_column, axis=1)
    y = pd.get_dummies(data[label_column])
    
    label_mapping = {index: label for index, label in enumerate(data[label_column].unique())}
    print(f'label_mapping is: {label_mapping}')

    # Split into train and test
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    # Select samples from each label in training data
    train_sampled = []
    for label in y.columns:
        x_label_train = x_train[y_train[label] == 1]
        y_label_train = y_train[y_train[label] == 1]
        available_samples = min(len(x_label_train), 13333)
        sampled_data = pd.concat([x_label_train, y_label_train], axis=1).sample(available_samples, random_state=42)
        print(f'Selected {available_samples} samples for label {label} in training data.')
        train_sampled.append(sampled_data)

    train_sampled_df = pd.concat(train_sampled)

    # Select samples from each label in testing data
    test_sampled = []
    for label in y.columns:
        x_label_test = x_test[y_test[label] == 1]
        y_label_test = y_test[y_test[label] == 1]
        available_samples = min(len(x_label_test),3333)
        sampled_data = pd.concat([x_label_test, y_label_test], axis=1).sample(available_samples, random_state=42)
        print(f'Selected {available_samples} samples for label {label} in testing data.')
        test_sampled.append(sampled_data)

    test_sampled_df = pd.concat(test_sampled)

    # Combine original and sampled data
    x_train_sampled = train_sampled_df.drop(columns=y.columns)
    y_train_sampled = train_sampled_df[y.columns]
    x_test_sampled = test_sampled_df.drop(columns=y.columns)
    y_test_sampled = test_sampled_df[y.columns]
    
    # Convert to NumPy arrays
    x_train = x_train.values
    y_train = y_train.values
    x_test = x_test.values
    y_test = y_test.values
    x_train_sampled = x_train_sampled.values
    y_train_sampled = y_train_sampled.values
    x_test_sampled = x_test_sampled.values
    y_test_sampled = y_test_sampled.values
    
    return x_train, x_test, y_train, y_test, x_train_sampled, y_train_sampled, x_test_sampled, y_test_sampled, label_mapping

            
def model(x_train_shape, y_train_shape, units, dropout, learning_rate,optimizer_name, activation):
    model = Sequential([
        LSTM(units,activation=activation,input_shape=(x_train_shape[1], x_train_shape[2])),
        Dropout(dropout),
        Dense(y_train_shape[1], activation='softmax')
    ])
    
    print(model.summary())
    
    optimizer = getattr(tf.keras.optimizers, optimizer_name)(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def calculate_metrics(y_test, y_pred):
    precision, recall, f1, _ = precision_recall_fscore_support(np.argmax(y_test, axis=1), np.argmax(y_pred, axis=1), average=None)
    overall_precision, overall_recall, overall_f1, _ = precision_recall_fscore_support(np.argmax(y_test, axis=1), np.argmax(y_pred, axis=1), average='macro')
    accuracy = accuracy_score(np.argmax(y_test, axis=1), np.argmax(y_pred, axis=1))
    return accuracy, precision, recall, f1, overall_precision, overall_recall, overall_f1




def log_metrics(run, y_test, y_pred, label_mapping):
    accuracy, precision, recall, f1, overall_precision, overall_recall, overall_f1 = calculate_metrics(y_test, y_pred)

#     run.log_metric('overall_accuracy', accuracy)
#     # print(f"overall_accuracy={accuracy};")
    run.log_metric('Final_precision', overall_precision)
#     # print(f"overall_precision={overall_precision};")
    run.log_metric('Final_recall', overall_recall)
#     # print(f"overall_recall={overall_recall};")
    run.log_metric('Final_f1_score', overall_f1)
#     # print(f"overall_f1_score={overall_f1};")
    
    
    
    for i, label in enumerate(label_mapping.values()):
        run.log_metric(f'Final_{label}_precision', precision[i])
        # print(f"{label}_precision={precision[i]};")
        run.log_metric(f'Final_{label}_recall', recall[i])
        # print(f"{label}_recall={recall[i]};")
        run.log_metric(f'Final_{label}_f1_score', f1[i])
        # print(f"{label}_f1_score={f1[i]};")
        

    # Compute confusion matrix
    cm = confusion_matrix(np.argmax(y_test, axis=1), np.argmax(y_pred, axis=1))
    # Generate confusion matrix heatmap
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=label_mapping.values(), yticklabels=label_mapping.values())

    cm_df = pd.DataFrame(cm, index=label_mapping.values(), columns=label_mapping.values())
    run.log_confusion_matrix(np.argmax(y_test, axis=1), np.argmax(y_pred, axis=1), "Confusion-Matrix-Test-Data")
#     run.log_precision_recall(np.argmax(y_test, axis=1), np.argmax(y_pred, axis=1), "log_precision_recall")
#     run.log_roc_curve(np.argmax(y_test, axis=1), np.argmax(y_pred, axis=1), "ROC Curve")
    
    
    

def upload_df_to_s3(df, bucket_name, object_key, s3_client):
    # Initialize a buffer for CSV content
    csv_buffer = StringIO()

    try:
        s3_client.head_object(Bucket=bucket_name, Key=object_key)
        file_exists = True
    except:
        file_exists = False
    

    if file_exists:
        # If the file exists, download its content and append the new data
        obj = s3_client.get_object(Bucket=bucket_name, Key=object_key)
        existing_df = pd.read_csv(obj['Body'])
        combined_df = pd.concat([existing_df, df], ignore_index=True)
        combined_df.to_csv(csv_buffer, index=False, header=True)
    else:
        # If the file does not exist, write the new DataFrame to the buffer
        df.to_csv(csv_buffer, index=False, header=True)

    # Upload the buffer content to S3, creating or updating the file
    s3_client.put_object(Body=csv_buffer.getvalue(), Bucket=bucket_name, Key=object_key)



################################## Single ##########################################

def calculate_shap_values(model, x_train, x_test):
    print("started SHAP")
    explainer = shap.GradientExplainer(model, x_train)
    shap_values = explainer.shap_values(x_test)
    print("Finished SHAP calculation")
    return shap_values


def shap_analysis_bar_plot_single_original(shap_values, x_test, bucket_name, folder_name, shap_single):
    try:
        print('SHAP analysis started')
        shap_values_2D = shap_values[0].reshape(-1, 16)
        x_test_2d = pd.DataFrame(x_test.reshape(-1, 16), columns=columns_features)
        explanation = shap.Explanation(values=shap_values_2D, data=x_test_2d, feature_names=columns_features)

        plt.figure(figsize=(10, 6))  # Set the figure size here
        shap.plots.bar(explanation, max_display=20)  # Remove the plot_size argument
        plt.title('Bar Plot')
        file_path = '/tmp/shap_summary_plot_original_bar_single_96.jpg'
        plt.savefig(file_path, bbox_inches='tight')  # Save with tight bounding box
        plt.close()

        # Upload to S3
        s3 = boto3.client('s3')
        s3_path = f'{folder_name}/{shap_single}/shap_summary_plot_original_bar_single_96.jpg'
        s3.upload_file(file_path, bucket_name, s3_path)

        print('SHAP analysis completed successfully')
    except Exception as e:
        print('Error during SHAP analysis:', str(e)) 

        
        
def shap_analysis_bar_plot_single(shap_values, x_test, bucket_name, folder_name, shap_single):
    try:
        print('SHAP analysis started')
        shap_values_2D = shap_values[0].reshape(-1, 16)
        x_test_2d = pd.DataFrame(x_test[:].reshape(-1, 16), columns=columns_features)
        explanation = shap.Explanation(values=shap_values_2D, data=x_test_2d, feature_names=columns_features)

        # Create summary plot with flipped axes
        plt.figure(figsize=(14, 12))  # Set the figure size to have enough space
        shap.summary_plot(shap_values_2D, x_test_2d, plot_type='bar', feature_names=columns_features, plot_size=None)

        plt.title('Bar Plot')
        file_path = '/tmp/shap_summary_plot_bar_single_96.jpg'
        plt.savefig(file_path)  # Save without tight bounding box to avoid cutting off labels
        plt.close()

        # Upload to S3
        s3 = boto3.client('s3')
        s3_path = f'{folder_name}/{shap_single}/shap_summary_plot_bar_single_96.jpg'
        s3.upload_file(file_path, bucket_name, s3_path)

        print('SHAP analysis completed successfully')
    except Exception as e:
        print('Error during SHAP analysis:', str(e))

    

        

def shap_analysis_violin_single(shap_values, x_test, bucket_name, folder_name, shap_single):
    try:
        print('SHAP analysis started')
        shap_values_2D = shap_values[0].reshape(-1, 16)
        x_test_2d = pd.DataFrame(x_test.reshape(-1, 16), columns=columns_features)
        explanation = shap.Explanation(values=shap_values_2D, data=x_test_2d, feature_names=columns_features)

        # Create summary plot
        plt.figure(figsize=(14, 6))
        plt.title('Layered Violin Summary Plot') 
        shap.summary_plot(shap_values_2D, features=x_test_2d, max_display=20, feature_names=columns_features, plot_type="violin", show=False)
        file_path = '/tmp/shap_violin_single_96.jpg'
        plt.savefig(file_path, bbox_inches='tight')
        plt.close()

        # Upload to S3
        s3 = boto3.client('s3')
        s3_path = f'{folder_name}/{shap_single}/shap_violin_single_96.jpg'
        s3.upload_file(file_path, bucket_name, s3_path)

        print('SHAP analysis completed successfully')
    except Exception as e:
        print('Error during SHAP analysis:', str(e))

          


def shap_analysis_summary_plot_single(shap_values, x_test, bucket_name, folder_name, shap_single):
    try:
        print('SHAP analysis started')
        shap_values_2D = shap_values[0].reshape(-1, 16)
        x_test_2d = pd.DataFrame(x_test.reshape(-1, 16), columns=columns_features)

        # Create summary plot
        plt.figure(figsize=(10, 6))
        plt.title('Summary Plot')  
        shap.summary_plot(shap_values_2D, x_test_2d)
        file_path = '/tmp/shap_summary_plot_single_96.jpg'
        plt.savefig(file_path)
        plt.close()

        # Upload to S3
        s3 = boto3.client('s3')
        s3_path = f'{folder_name}/{shap_single}/shap_summary_plot_single_96.jpg'
        s3.upload_file(file_path, bucket_name, s3_path)

        print('SHAP analysis completed successfully')
    except Exception as e:
        print('Error during SHAP analysis:', str(e))



              

def shap_analysis_beeswarm_single(shap_values, x_test, bucket_name, folder_name, shap_single):
    try:
        print('SHAP analysis started')
        shap_values_2D = shap_values[0].reshape(-1, 16)
        x_test_2d = pd.DataFrame(x_test.reshape(-1, 16), columns=columns_features)
        explanation = shap.Explanation(values=shap_values_2D, data=x_test_2d, feature_names=columns_features)

        # Create summary plot
        plt.figure(figsize=(10, 6))
        plt.title('Beeswarm Summary Plot') 
        shap.plots.beeswarm(explanation, max_display=20)
        file_path = '/tmp/shap_beeswarm_single_96.jpg'
        plt.savefig(file_path, bbox_inches='tight')
        plt.close()

        # Upload to S3
        s3 = boto3.client('s3')
        s3_path = f'{folder_name}/{shap_single}/shap_beeswarm_single_96.jpg'
        s3.upload_file(file_path, bucket_name, s3_path)

        print('SHAP analysis completed successfully')
    except Exception as e:
        print('Error during SHAP analysis:', str(e)) 




def shap_analysis_violin_layered_single(shap_values, x_test, bucket_name, folder_name, shap_single):
    try:
        print('SHAP analysis started')
        shap_values_2D = shap_values[0].reshape(-1, 16)
        x_test_2d = pd.DataFrame(x_test.reshape(-1, 16), columns=columns_features)
        explanation = shap.Explanation(values=shap_values_2D, data=x_test_2d, feature_names=columns_features)

        # Create bar plot for each class
        plt.figure(figsize=(14,8))
        shap.summary_plot(shap_values_2D, features=x_test_2d, feature_names=columns_features, max_display=20, plot_type="layered_violin", show=False)
        plt.title('Layered Violin Summary Plot')
        file_path = '/tmp/shap_violin_layered_single_96.jpg'
        plt.savefig(file_path, bbox_inches='tight')
        plt.close()

        # Upload to S3
        s3 = boto3.client('s3')
        s3_path = f'{folder_name}/{shap_single}/shap_violin_layered_single_96.jpg'
        s3.upload_file(file_path, bucket_name, s3_path)

        print('SHAP analysis completed successfully')
    except Exception as e:
        print('Error during SHAP analysis:', str(e))   


def shap_analysis_heatmap_single(shap_values, x_test, bucket_name, folder_name, shap_single):
    try:
        print('SHAP analysis started')
        shap_values_2D = shap_values[0].reshape(-1, 16)
        x_test_2d = pd.DataFrame(x_test.reshape(-1, 16), columns=columns_features)
        explanation = shap.Explanation(values=shap_values_2D, data=x_test_2d, feature_names=columns_features)

        # Create summary plot
        plt.figure(figsize=(10, 6))
        plt.title('Heatmap Summary Plot') 
        shap.plots.heatmap(explanation, max_display=20)
        file_path = '/tmp/shap_heatmap_single_96.jpg'
        plt.savefig(file_path, bbox_inches='tight')
        plt.close()

        # Upload to S3
        s3 = boto3.client('s3')
        s3_path = f'{folder_name}/{shap_single}/shap_heatmap_single_96.jpg'
        s3.upload_file(file_path, bucket_name, s3_path)

        print('SHAP analysis completed successfully')
    except Exception as e:
        print('Error during SHAP analysis:', str(e))


         
################################# Multiclass ###################################

def shap_analysis_bar_multiclass_original(shap_values, x_test, bucket_name, folder_name, shap_multi, label_mapping):
    try:
        print('SHAP analysis started')
        print('Bar started')
        
        for i, shap_value in enumerate(shap_values):
            shap_values_2D = shap_value.reshape(-1, 16)
            x_test_2d = pd.DataFrame(x_test.reshape(-1, 16), columns=columns_features)
            explanation = shap.Explanation(values=shap_values_2D, data=x_test_2d, feature_names=columns_features)

            plt.figure(figsize=(10, 6))
            shap.plots.bar(explanation, max_display=20)
            plt.title(f'Bar Plot - {label_mapping.get(i, f"UnknownLabel{i}")}')
            file_path = f'/tmp/shap_bar_plot_multiclass_{label_mapping[i]}_original_96.jpg'
            plt.savefig(file_path, bbox_inches='tight')
            plt.close()

            s3 = boto3.client('s3')
            s3_path = f'{folder_name}/{shap_multi}/shap_bar_plot_multiclass_{label_mapping[i]}_original_96.jpg'
            s3.upload_file(file_path, bucket_name, s3_path)

        print('SHAP analysis completed successfully')
    except Exception as e:
        print('Error during SHAP analysis:', str(e))


def shap_analysis_bar_multiclass(shap_values, x_test, bucket_name, folder_name, shap_multi, label_mapping):
    try:
        print('SHAP analysis started')
        
        for i, shap_value in enumerate(shap_values):
            shap_values_2D = shap_value.reshape(-1, 16)
            x_test_2d = pd.DataFrame(x_test.reshape(-1, 16), columns=columns_features)

            plt.figure(figsize=(17, 14))
            shap.summary_plot(shap_values_2D, x_test_2d, plot_type="bar", plot_size=None, show=False)
            plt.title(f'Bar Plot - {label_mapping.get(i, f"UnknownLabel{i}")}')
            file_path = f'/tmp/shap_bar_plot_multiclass_{label_mapping[i]}_96.jpg'
            plt.savefig(file_path, bbox_inches='tight')
            plt.close()

            s3 = boto3.client('s3')
            s3_path = f'{folder_name}/{shap_multi}/shap_bar_plot_multiclass_{label_mapping[i]}_96.jpg'
            s3.upload_file(file_path, bucket_name, s3_path)

        print('SHAP analysis completed successfully')
    except Exception as e:
        print('Error during SHAP analysis:', str(e))


def shap_analysis_summary_plot_multiclass(shap_values, x_test, bucket_name, folder_name, shap_multi, label_mapping):
    try:
        print('SHAP analysis started')

        for i, shap_value in enumerate(shap_values):
            shap_values_2D = shap_value.reshape(-1, 16)
            x_test_2d = pd.DataFrame(x_test.reshape(-1, 16), columns=columns_features)

            plt.figure(figsize=(10, 6))
            shap.summary_plot(shap_values_2D, x_test_2d, show=False)
            plt.title(f'Summary Plot - {label_mapping.get(i, f"UnknownLabel{i}")}')
            file_path = f'/tmp/shap_summary_plot_multiclass_{label_mapping[i]}_96.jpg'
            plt.savefig(file_path, bbox_inches='tight')
            plt.close()

            s3 = boto3.client('s3')
            s3_path = f'{folder_name}/{shap_multi}/shap_summary_plot_multiclass_{label_mapping[i]}_96.jpg'
            s3.upload_file(file_path, bucket_name, s3_path)

        print('SHAP analysis completed successfully')
    except Exception as e:
        print('Error during SHAP analysis:', str(e))


def shap_analysis_beeswarm_multiclass(shap_values, x_test, bucket_name, folder_name, shap_multi, label_mapping):
    try:
        print('SHAP analysis started')

        for i, shap_value in enumerate(shap_values):
            shap_values_2D = shap_value.reshape(-1, 16)
            x_test_2d = pd.DataFrame(x_test.reshape(-1, 16), columns=columns_features)
            explanation = shap.Explanation(values=shap_values_2D, data=x_test_2d, feature_names=columns_features)

            plt.figure(figsize=(10, 6))
            shap.plots.beeswarm(explanation, max_display=20)
            plt.title(f'Beeswarm Summary Plot - {label_mapping.get(i, f"UnknownLabel{i}")}')
            file_path = f'/tmp/shap_beeswarm_multiclass_{label_mapping[i]}_96.jpg'
            plt.savefig(file_path, bbox_inches='tight')
            plt.close()

            s3 = boto3.client('s3')
            s3_path = f'{folder_name}/{shap_multi}/shap_beeswarm_multiclass_{label_mapping[i]}_96.jpg'
            s3.upload_file(file_path, bucket_name, s3_path)

        print('SHAP analysis completed successfully')
    except Exception as e:
        print('Error during SHAP analysis:', str(e))


def shap_analysis_violin_multiclass(shap_values, x_test, bucket_name, folder_name, shap_multi, label_mapping):
    try:
        print('SHAP analysis started')

        for i, shap_value in enumerate(shap_values):
            shap_values_2D = shap_value.reshape(-1, 16)
            x_test_2d = pd.DataFrame(x_test.reshape(-1, 16), columns=columns_features)
            explanation = shap.Explanation(values=shap_values_2D, data=x_test_2d, feature_names=columns_features)

            plt.figure(figsize=(14,8))
            shap.summary_plot(shap_values_2D, features=x_test_2d, feature_names=columns_features, max_display=20, plot_type="violin", show=False)
            plt.title(f'Violin Summary Plot - {label_mapping.get(i, f"UnknownLabel{i}")}')
            file_path = f'/tmp/shap_violin_multiclass_{label_mapping[i]}_96.jpg'
            plt.savefig(file_path, bbox_inches='tight')
            plt.close()

            s3 = boto3.client('s3')
            s3_path = f'{folder_name}/{shap_multi}/shap_violin_multiclass_{label_mapping[i]}_96.jpg'
            s3.upload_file(file_path, bucket_name, s3_path)

        print('SHAP analysis completed successfully')
    except Exception as e:
        print('Error during SHAP analysis:', str(e))


def shap_analysis_violin_layered_multiclass(shap_values, x_test, bucket_name, folder_name, shap_multi, label_mapping):
    try:
        print('SHAP analysis started')

        for i, shap_value in enumerate(shap_values):
            shap_values_2D = shap_value.reshape(-1, 16)
            x_test_2d = pd.DataFrame(x_test.reshape(-1, 16), columns=columns_features)
            explanation = shap.Explanation(values=shap_values_2D, data=x_test_2d, feature_names=columns_features)

            plt.figure(figsize=(14,8))
            shap.summary_plot(shap_values_2D, features=x_test_2d, feature_names=columns_features, max_display=20, plot_type="layered_violin", show=False)
            plt.title(f'Layered Violin Summary Plot - {label_mapping.get(i, f"UnknownLabel{i}")}')
            file_path = f'/tmp/shap_violin_layered_multiclass_{label_mapping[i]}_96.jpg'
            plt.savefig(file_path, bbox_inches='tight')
            plt.close()

            s3 = boto3.client('s3')
            s3_path = f'{folder_name}/{shap_multi}/shap_violin_layered_multiclass_{label_mapping[i]}_96.jpg'
            s3.upload_file(file_path, bucket_name, s3_path)

        print('SHAP analysis completed successfully')
    except Exception as e:
        print('Error during SHAP analysis:', str(e))


def shap_analysis_heatmap_multiclass(shap_values, x_test, bucket_name, folder_name, shap_multi, label_mapping):
    try:
        print('SHAP analysis started')

        for i, shap_value in enumerate(shap_values):
            shap_values_2D = shap_value.reshape(-1, 16)
            x_test_2d = pd.DataFrame(x_test.reshape(-1, 16), columns=columns_features)
            explanation = shap.Explanation(values=shap_values_2D, data=x_test_2d, feature_names=columns_features)

            plt.figure(figsize=(10, 6))
            shap.plots.heatmap(explanation, max_display=20)
            plt.title(f'Heatmap Summary Plot - {label_mapping.get(i, f"UnknownLabel{i}")}')
            file_path = f'/tmp/shap_heatmap_multiclass_{label_mapping[i]}_96.jpg'
            plt.savefig(file_path, bbox_inches='tight')
            plt.close()

            s3 = boto3.client('s3')
            s3_path = f'{folder_name}/{shap_multi}/shap_heatmap_multiclass_{label_mapping[i]}_96.jpg'
            s3.upload_file(file_path, bucket_name, s3_path)

        print('SHAP analysis completed successfully')
    except Exception as e:
        print('Error during SHAP analysis:', str(e))


def shap_analysis_grouped_bar_multiclass_vertical(shap_values, x_test, bucket_name, folder_name, shap_grouped, label_mapping):
    try:
        custom_colors = ['#2C4E80', '#D20062', '#2C7865', '#F99417', '#B15EFF', '#0081B4']
        if not isinstance(label_mapping, dict):
            raise ValueError("label_mapping must be a dictionary")

        print('SHAP analysis started')

        feature_importance_df = pd.DataFrame(index=label_mapping.values(), columns=columns_features)

        for i, shap_array in enumerate(shap_values):
            label = label_mapping[i]
            feature_importance_df.loc[label] = np.abs(shap_array).mean(axis=0)

        ax = feature_importance_df.T.plot(kind='bar', figsize=(16, 12), width=0.6, color=custom_colors)
        ax.set_title('Feature Importance by Label')
        ax.set_ylabel('Mean |SHAP value|')
        plt.xticks(rotation=90)
        plt.legend(title='Label')

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        file_path = '/tmp/shap_grouped_bar_plot_vertical_96.jpg'
        plt.tight_layout()
        plt.savefig(file_path)
        plt.close()

        s3 = boto3.client('s3')
        s3_path = f'{folder_name}/{shap_grouped}/shap_grouped_bar_plot_vertical_96.jpg'
        s3.upload_file(file_path, bucket_name, s3_path)

        print('SHAP analysis completed successfully')
    except Exception as e:
        print('Error during SHAP analysis:', str(e))


def shap_analysis_grouped_bar_multiclass_horizontal(shap_values, x_test, bucket_name, folder_name, shap_grouped, label_mapping):
    try:
        custom_colors = ['#2C4E80', '#D20062', '#2C7865', '#F99417', '#B15EFF', '#0081B4']
        if not isinstance(label_mapping, dict):
            raise ValueError("label_mapping must be a dictionary")

        print('SHAP analysis started')

        feature_importance_df = pd.DataFrame(index=label_mapping.values(), columns=columns_features)

        for i, shap_array in enumerate(shap_values):
            label = label_mapping[i]
            feature_importance_df.loc[label] = np.abs(shap_array).mean(axis=0)

        ax = feature_importance_df.T.plot(kind='barh', figsize=(16, 12), width=0.6, color=custom_colors)
        ax.set_title('Feature Importance by Label')
        ax.set_xlabel('Mean |SHAP value|')
        plt.yticks(rotation=0)
        plt.legend(title='Label')

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        file_path = '/tmp/shap_grouped_bar_plot_horizontal_96.jpg'
        plt.tight_layout()
        plt.savefig(file_path)
        plt.close()

        s3 = boto3.client('s3')
        s3_path = f'{folder_name}/{shap_grouped}/shap_grouped_bar_plot_horizontal_96.jpg'
        s3.upload_file(file_path, bucket_name, s3_path)

        print('SHAP analysis completed successfully')
    except Exception as e:
        print('Error during SHAP analysis:', str(e))


def shap_analysis_violin_multiclass_stick_with_seaborn(shap_values, x_test, bucket_name, folder_name, shap_violin, label_mapping):
    try:
        print('Violin started')
        custom_colors = ['#2C4E80', '#D20062', '#2C7865', '#F99417', '#B15EFF', '#0081B4']

        print('SHAP analysis started')

        assert len(custom_colors) >= len(shap_values), "Not enough colors provided for each class."

        for i, shap_value in enumerate(shap_values):
            shap_values_2D = shap_value.reshape(-1, len(columns_features))
            x_test_2d = pd.DataFrame(x_test.reshape(-1, len(columns_features)), columns=columns_features)

            df_long = pd.DataFrame(shap_values_2D, columns=columns_features)
            df_long = df_long.melt(var_name='Feature', value_name='SHAP Value')

            plt.figure(figsize=(10, 6))
            sns.set(style="whitegrid")
            sns.violinplot(x='Feature', y='SHAP Value', data=df_long, scale='width', palette=[custom_colors[i]])
            
            plt.title(f'Violin Summary Plot - {label_mapping.get(i, f"UnknownLabel{i}")}')
            plt.xticks(rotation=90)

            file_path = f'/tmp/shap_violin_multiclass_stick_{label_mapping.get(i, f"UnknownLabel{i}")}_96.jpg'
            plt.savefig(file_path, bbox_inches='tight')
            plt.close()

            s3 = boto3.client('s3')
            s3_path = f'{folder_name}/{shap_violin}/shap_violin_multiclass_stick_{label_mapping.get(i, f"UnknownLabel{i}")}_96.jpg'
            s3.upload_file(file_path, bucket_name, s3_path)

        print('SHAP analysis completed successfully')
    except Exception as e:
        print('Error during SHAP analysis:', str(e))


def shap_analysis_violin_layered_single_stick_with_seaborn(shap_values, x_test, bucket_name, folder_name, shap_violin, label_mapping):
    try:
        print('SHAP analysis started')

        shap_values_2D = shap_values[0].reshape(-1, len(columns_features))
        x_test_2d = pd.DataFrame(x_test.reshape(-1, len(columns_features)), columns=columns_features)

        df_long = pd.DataFrame(shap_values_2D, columns=columns_features)
        df_long = df_long.melt(var_name='Feature', value_name='SHAP Value')

        plt.figure(figsize=(12, 8))
        sns.set(style="whitegrid")
        sns.violinplot(x='Feature', y='SHAP Value', data=df_long, scale='width')
        plt.title('Violin Summary Plot')
        plt.xticks(rotation=90)

        file_path = '/tmp/shap_violin_horizontal_seaborn_96.jpg'
        plt.savefig(file_path, bbox_inches='tight')
        plt.close()

        s3 = boto3.client('s3')
        s3_path = f'{folder_name}/{shap_violin}/shap_violin_horizontal_seaborn_96.jpg'
        s3.upload_file(file_path, bucket_name, s3_path)

        print('SHAP analysis completed successfully')
    except Exception as e:
        print('Error during SHAP analysis:', str(e))


def shap_analysis_violin_multiclass_with_seaborn(shap_values, x_test, bucket_name, folder_name, shap_violin, label_mapping):
    try:
        custom_colors = ['#2C4E80', '#D20062', '#2C7865', '#F99417', '#B15EFF', '#0081B4']

        print('SHAP analysis started')

        for i, shap_value in enumerate(shap_values):
            shap_values_2D = shap_value.reshape(-1, len(columns_features))
            x_test_2d = pd.DataFrame(x_test.reshape(-1, len(columns_features)), columns=columns_features)

            df_long = pd.DataFrame(shap_values_2D, columns=columns_features)
            df_long = df_long.melt(var_name='Feature', value_name='SHAP Value')

            plt.figure(figsize=(10, 6))
            sns.set(style="whitegrid")
            sns.violinplot(x='Feature', y='SHAP Value', data=df_long, scale='width', inner='quartile', palette=[custom_colors[i]])
            plt.title(f'Violin Summary Plot - {label_mapping.get(i, f"UnknownLabel{i}")}')
            plt.xticks(rotation=90)

            file_path = f'/tmp/shap_violin_multiclass_{label_mapping.get(i, f"UnknownLabel{i}")}_96.jpg'
            plt.savefig(file_path, bbox_inches='tight')
            plt.close()

            s3 = boto3.client('s3')
            s3_path = f'{folder_name}/{shap_violin}/shap_violin_multiclass_{label_mapping.get(i, f"UnknownLabel{i}")}_96.jpg'
            s3.upload_file(file_path, bucket_name, s3_path)

        print('SHAP analysis completed successfully')
    except Exception as e:
        print('Error during SHAP analysis:', str(e))


def shap_analysis_violin_layered_single_with_seaborn(shap_values, x_test, bucket_name, folder_name, shap_violin, label_mapping):
    try:
        print('SHAP analysis started')

        shap_values_2D = shap_values[0].reshape(-1, len(columns_features))
        x_test_2d = pd.DataFrame(x_test.reshape(-1, len(columns_features)), columns=columns_features)

        df_long = pd.DataFrame(shap_values_2D, columns=columns_features)
        df_long = df_long.melt(var_name='Feature', value_name='SHAP Value')

        plt.figure(figsize=(12, 8))
        sns.set(style="whitegrid")
        sns.violinplot(x='Feature', y='SHAP Value', data=df_long, scale='width', inner='quartile')
        plt.title('Violin Summary Plot')
        plt.xticks(rotation=90)

        file_path = '/tmp/shap_violin_horizontal_seaborn_96.jpg'
        plt.savefig(file_path, bbox_inches='tight')
        plt.close()

        s3 = boto3.client('s3')
        s3_path = f'{folder_name}/{shap_violin}/shap_violin_horizontal_seaborn_96.jpg'
        s3.upload_file(file_path, bucket_name, s3_path)

        print('SHAP analysis completed successfully')
    except Exception as e:
        print('Error during SHAP analysis:', str(e))


        

        
def main():
    
    args = parse_args()
    bucket_name = 'multiclass-balanced-two-million'
    folder_name = "SHAP-decrease-main-model-performance-experiments-96"  # Name of the folder where you want to save the files
    shap_multi = 'shap/multi'
    shap_single = 'shap/single'
    shap_grouped='shap/grouped'
    shap_violin='shap/violin'
    
    
    # run_name = get_experiment_config()  # Get the run name from the experiment config
    # run_name = f"DefaultRunName-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    # x_train, x_test, y_train, y_test, label_mapping = load_data(args.label_column)
    x_train, x_test, y_train, y_test, x_train_sampled, y_train_sampled, x_test_sampled, y_test_sampled,label_mapping =load_data(args.label_column)
    x_train = x_train.reshape((x_train.shape[0], 1, x_train.shape[1]))
    x_test = x_test.reshape((x_test.shape[0], 1, x_test.shape[1]))
    x_train_sampled = x_train_sampled.reshape((x_train_sampled.shape[0], 1, x_train_sampled.shape[1]))
    x_test_sampled = x_test_sampled.reshape((x_test_sampled.shape[0], 1, x_test_sampled.shape[1]))
    
    
    strategy = tf.distribute.MirroredStrategy()
    
    # lstm_model = model(x_train.shape, y_train.shape, args.units, args.dropout,args.learning_rate,args.optimizer,args.activation)
    print(f"Number of devices: {strategy.num_replicas_in_sync}")
    
    # with strategy.scope():
    lstm_model = model(x_train.shape, y_train.shape, args.units, args.dropout, args.learning_rate, args.optimizer, args.activation)

    batch_size = args.batch_size
    epochs = args.epochs

    with load_run(sagemaker_session=sagemaker_session) as run:
    # with Run(experiment_name=experiment_name,run_name=run_name) as run:
        run.log_parameters({
            'units': args.units,
            'learning_rate': args.learning_rate,
            'dropout': args.dropout,
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'optimizer':args.optimizer,
            'activation':args.activation
            
        })
        


        history = lstm_model.fit(
            x_train, y_train,
            epochs=args.epochs,
            batch_size=args.batch_size,
            validation_data=(x_test, y_test),  # Provide validation data here
            callbacks=[ExperimentCallback(run, lstm_model, x_test, y_test)]
        )
        
     
        y_pred = lstm_model.predict(x_test)
        
        
         ##########################Explainability Learning#####################
        # shap_values = calculate_shap_values(lstm_model, x_train, x_test)
        with strategy.scope():
            shap_values = calculate_shap_values(lstm_model, x_train_sampled, x_test_sampled)
#         #single

#         shap_analysis_bar_plot_single_original(lstm_model, x_train, x_test,bucket_name, folder_name,shap_single)
#         shap_analysis_bar_plot_single(lstm_model, x_train, x_test,bucket_name, folder_name,shap_single)
#         shap_analysis_summary_plot_single(lstm_model, x_train, x_test,bucket_name, folder_name,shap_single)
#         shap_analysis_beeswarm_single(lstm_model, x_train, x_test,bucket_name, folder_name,shap_single)
#         shap_analysis_violin_single(lstm_model, x_train, x_test,bucket_name, folder_name,shap_single)
#         shap_analysis_violin_layered_single(lstm_model, x_train, x_test,bucket_name, folder_name,shap_single)
#         shap_analysis_heatmap_single(lstm_model, x_train, x_test,bucket_name, folder_name,shap_single)
        
        
#         #multiclass
        
#         shap_analysis_bar_multiclass_original(lstm_model, x_train, x_test,bucket_name,folder_name,shap_multi,label_mapping)
#         shap_analysis_bar_multiclass(lstm_model, x_train, x_test,bucket_name,folder_name,shap_multi,label_mapping)
#         shap_analysis_summary_plot_multiclass(lstm_model, x_train, x_test,bucket_name,folder_name,shap_multi, label_mapping)
#         shap_analysis_beeswarm_multiclass(lstm_model, x_train, x_test,bucket_name, folder_name,shap_multi,label_mapping)
#         shap_analysis_violin_multiclass(lstm_model, x_train, x_test,bucket_name, folder_name,shap_multi, label_mapping)
#         shap_analysis_violin_layered_multiclass(lstm_model, x_train,x_test,bucket_name,folder_name,shap_multi,label_mapping)
#         shap_analysis_heatmap_multiclass(lstm_model, x_train, x_test,bucket_name,folder_name,shap_multi,label_mapping)
        
        
#         #grouped
#         shap_analysis_grouped_bar_multiclass_vertical(lstm_model, x_train, x_test, bucket_name, folder_name, shap_grouped, label_mapping)
#         shap_analysis_grouped_bar_multiclass_horizontal(lstm_model, x_train, x_test, bucket_name, folder_name, shap_grouped, label_mapping)
        
        
#         #violin-modified
#         shap_analysis_violin_layered_single_with_seaborn(lstm_model, x_train, x_test, bucket_name, folder_name, shap_violin, label_mapping)
#         shap_analysis_violin_multiclass_with_seaborn(lstm_model, x_train, x_test, bucket_name, folder_name, shap_violin, label_mapping)
#         shap_analysis_violin_multiclass_stick_with_seaborn(lstm_model,x_train, x_test, bucket_name, folder_name, shap_violin, label_mapping)
#         shap_analysis_violin_layered_single_stick_with_seaborn(lstm_model, x_train, x_test, bucket_name, folder_name, shap_violin, label_mapping)

        #single
        # shap_analysis_bar_plot_single_original(shap_values, x_test_sampled, bucket_name, folder_name, shap_single)
        # shap_analysis_bar_plot_single(shap_values, x_test_sampled, bucket_name, folder_name, shap_single)
        # shap_analysis_summary_plot_single(shap_values, x_test_sampled, bucket_name, folder_name, shap_single)
        # shap_analysis_beeswarm_single(shap_values, x_test_sampled, bucket_name, folder_name, shap_single)
        # shap_analysis_violin_single(shap_values, x_test_sampled, bucket_name, folder_name, shap_single)
        # shap_analysis_violin_layered_single(shap_values,  x_test_sampled, bucket_name, folder_name, shap_single)
        
        #multiclass
        shap_analysis_bar_multiclass_original(shap_values, x_test_sampled, bucket_name, folder_name, shap_multi, label_mapping)
        shap_analysis_bar_multiclass(shap_values, x_test_sampled, bucket_name, folder_name, shap_multi, label_mapping)
        
        #violin-modified
        shap_analysis_violin_multiclass_stick_with_seaborn(shap_values, x_test_sampled, bucket_name, folder_name, shap_violin, label_mapping)
        shap_analysis_violin_layered_single_stick_with_seaborn(shap_values, x_test_sampled, bucket_name, folder_name, shap_violin, label_mapping)
        
        # shap_analysis_violin_layered_single_with_seaborn(shap_values, x_test_sampled, bucket_name, folder_name, shap_violin, label_mapping)
        # shap_analysis_violin_multiclass_with_seaborn(shap_values, x_test_sampled, bucket_name, folder_name, shap_violin, label_mapping)
        
        #########
        #grouped
        shap_analysis_grouped_bar_multiclass_vertical(shap_values, x_test_sampled, bucket_name, folder_name, shap_grouped, label_mapping)
        shap_analysis_grouped_bar_multiclass_horizontal(shap_values, x_test_sampled, bucket_name, folder_name, shap_grouped, label_mapping)
        shap_analysis_violin_layered_multiclass(shap_values, x_test_sampled, bucket_name, folder_name, shap_multi, label_mapping)
        shap_analysis_violin_multiclass(shap_values, x_test_sampled, bucket_name, folder_name, shap_multi, label_mapping)
        ###########
        shap_analysis_summary_plot_multiclass(shap_values, x_test_sampled, bucket_name, folder_name, shap_multi, label_mapping)
        shap_analysis_beeswarm_multiclass(shap_values, x_test_sampled, bucket_name, folder_name, shap_multi, label_mapping)
       
        
       
        
        ##########################Performance Evaluation######################
        
        train_loss, train_accuracy = history.history['loss'][-1], history.history['accuracy'][-1]
        test_loss, test_accuracy = history.history['val_loss'][-1], history.history['val_accuracy'][-1]

        run.log_metric('train:loss', train_loss,step=epochs)
        run.log_metric('Final_train_loss', train_loss)
        

        run.log_metric('train: accuracy', train_accuracy,step=epochs)
        run.log_metric('Final_train_accuracy', train_accuracy)


        # run.log_metric('test: loss', test_loss,step=epochs)
        run.log_metric('test: loss', test_loss,step=epochs)
        run.log_metric('Final_test_loss', test_loss)
       

        run.log_metric('test: accuracy', test_accuracy,step=epochs)
        run.log_metric('Final_test_accuracy', test_accuracy)


        # Predict with the model


        # Calculate metrics
        accuracy, precision, recall, f1, overall_precision, overall_recall, overall_f1 = calculate_metrics(y_test, y_pred)

        # Log metrics (make sure this function exists and is adjusted as needed)
        log_metrics(run, y_test, y_pred, label_mapping)


        # Prepare hyperparameters
        hyperparameters = {
            'Units': args.units,
            'Learning Rate': args.learning_rate,
            'Dropout': args.dropout,
            'Epochs': args.epochs,
            'Batch Size': args.batch_size, 
            'Optimizer':args.optimizer,
            'Activation':args.activation
              
        }

        # Detailed metrics for each label
        detailed_metrics = [{'Label': label_mapping.get(i, f'UnknownLabel{i}'), 'Precision': p, 'Recall': r, 'F1 Score': f} for i, (p, r, f) in enumerate(zip(precision, recall, f1))]
        for metric in detailed_metrics:
            metric.update(hyperparameters)
        detailed_metrics_df = pd.DataFrame(detailed_metrics)
 

        # Overall metrics
        overall_metrics = {
            'Overall Accuracy': [accuracy],
            'Overall Precision': [overall_precision],
            'Overall Recall': [overall_recall],
            'Overall F1 Score': [overall_f1]
        }


        overall_metrics.update({k: [v] for k, v in hyperparameters.items()})
        overall_metrics_df = pd.DataFrame(overall_metrics)


        # S3 Details

        # bucket_name = 'multiclass-balanced-two-million'
        # folder_name = 'LSTM-Shap'  # Name of the folder where you want to save the files
        overall_metrics_key = f'{folder_name}/compare/overall-metrics-lstm.csv'  # Updated key with folder name
        detailed_metrics_key = f'{folder_name}/compare/detailed-metrics-lstm.csv'  # Updated key with folder name


        ## Assuming 's3' is your boto3 S3 client
        upload_df_to_s3(overall_metrics_df, bucket_name, overall_metrics_key, s3)
        upload_df_to_s3(detailed_metrics_df, bucket_name, detailed_metrics_key, s3)

        # Save the model

        lstm_model.save(os.path.join(args.model_dir, 'model'), save_format='tf')

            
if __name__ == '__main__':
    main()

