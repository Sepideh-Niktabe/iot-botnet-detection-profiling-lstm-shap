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

import pandas as pd
import boto3
from io import StringIO
import botocore
import os
import json

# Use the same experiment name as defined in the training script
experiment_name =  "mainhyperparametertuningoutput"

# Set the AWS region
aws_region = os.environ.get('AWS_REGION', 'us-east-2') 
boto_session= boto3.setup_default_session(region_name=aws_region)
sagemaker_session = Session(boto_session)
s3 = boto3.client("s3")






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
    parser.add_argument('--trial_name', type=str, required=True)  # Accept trial name as an argument
    ##########
    parser.add_argument('--optimizer', type=str, default='Adam')
    parser.add_argument('--activation', type=str, default='tanh')
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
    #TODO: write bucket and file_key as variables
    #TODO: select columns
    columns = ['mode_payload_bytes_delta_len', 'mode_fwd_packets_delta_len',
               'mean_bwd_payload_bytes_delta_len', 'cov_bwd_header_bytes_delta_len',
               'mode_packets_delta_len', 'cov_bwd_packets_delta_len', 'active_skewness',
               'median_bwd_packets_delta_len', 'bwd_payload_bytes_skewness',
               'mean_bwd_packets_delta_len', 'mode_bwd_payload_bytes_delta_len',
               'fwd_total_header_bytes', 'median_fwd_payload_bytes_delta_len',
               'skewness_bwd_header_bytes_delta_len', 'active_min',
               'mode_fwd_payload_bytes_delta_len', 'Sample Classification']
    
    bucket = 'multiclass-balanced-two-million'
    file_key = 'merged-multiclass-balanced-two-million.csv'
    s3_client = boto3.client('s3')
    csv_obj = s3_client.get_object(Bucket=bucket, Key=file_key )
    body = csv_obj['Body']
    csv_string = body.read().decode('utf-8')
    data = pd.read_csv(io.StringIO(csv_string),usecols=columns)
    x = data.drop(label_column, axis=1).values
    y = pd.get_dummies(data[label_column]).values
    label_mapping = {index: label for index, label in enumerate(data[label_column].unique())}
    x_train, x_test, y_train, y_test=train_test_split(x, y, test_size=0.2, random_state=42)
    
    return  x_train, x_test, y_train, y_test, label_mapping
           
      
            
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


def main():
    
    args = parse_args()
    # run_name = get_experiment_config()  # Get the run name from the experiment config
    run_name = f"DefaultRunName-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    x_train, x_test, y_train, y_test, label_mapping = load_data(args.label_column)
    x_train = x_train.reshape((x_train.shape[0], 1, x_train.shape[1]))
    x_test = x_test.reshape((x_test.shape[0], 1, x_test.shape[1]))
    
    lstm_model = model(x_train.shape, y_train.shape, args.units, args.dropout,args.learning_rate,args.optimizer,args.activation)


    batch_size = args.batch_size
    epochs = args.epochs

    # with load_run(experiment_name=experiment_name, run_name=args.trial_name) as run:
    with Run(experiment_name=experiment_name,run_name=run_name) as run:
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
            callbacks=[ExperimentCallback(run, model, x_test, y_test)]
        )



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
        y_pred = lstm_model.predict(x_test)

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
        # detailed_metrics_df.to_csv('detailed_metrics.csv', index=False)
        # detailed_metrics_df.to_csv(os.path.join(args.model_dir, 'detailed_metrics.csv'), index=False)

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

        bucket_name = 'multiclass-balanced-two-million'
        folder_name = 'main-lstm-ouput'  # Name of the folder where you want to save the files
        overall_metrics_key = f'{folder_name}/overall-metrics.csv'  # Updated key with folder name
        detailed_metrics_key = f'{folder_name}/detailed-metrics.csv'  # Updated key with folder name


        ## Assuming 's3' is your boto3 S3 client
        upload_df_to_s3(overall_metrics_df, bucket_name, overall_metrics_key, s3)
        upload_df_to_s3(detailed_metrics_df, bucket_name, detailed_metrics_key, s3)

        # Save the model

        lstm_model.save(os.path.join(args.model_dir, 'model'), save_format='tf')

            
if __name__ == '__main__':
    main()

