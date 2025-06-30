import os
import argparse
import numpy as np
import pandas as pd
from datasets import load_dataset, Dataset, DatasetDict
from transformers import (
    DistilBertTokenizer,
    DistilBertForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

def preprocess_function(examples, tokenizer, max_length=512):
    return tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=max_length
    )

def load_datasets(data_dir='data'):
    """
    Load all CSV files from the data directory into a dictionary of dataframes.
    
    Args:
        data_dir (str): Path to the directory containing the CSV files
        
    Returns:
        dict: Dictionary with filenames as keys and corresponding dataframes as values
    """
    datasets = {}
    
    # List all CSV files in the data directory
    csv_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
    
    for file in csv_files:
        print(f'Loading {file}...')
        file_path = os.path.join(data_dir, file)
        
        # Load the dataframe
        df = pd.read_csv(file_path)
            
        # Store in dictionary using filename without extension as key
        key = file.replace('.csv', '')
        datasets[key] = df[['message', 'label']]
    
    return datasets

def prepare_balanced_datasets(datasets_dict, train_size=1000, test_size=500):
    """
    Prepare balanced training and testing datasets from multiple dataframes.
    
    Args:
        datasets_dict (dict): Dictionary of dataframes
        train_size (int): Number of samples per class for training
        test_size (int): Number of samples per class for testing
        
    Returns:
        tuple: (combined_train_df, combined_test_df)
    """
    all_train_dfs = []
    all_test_dfs = []
    
    for dataset_name, df in datasets_dict.items():
        # Get unique labels
        labels = df['label'].unique()
        
        train_samples = []
        test_samples = []
        
        for label in labels:
            # Get all samples for this label
            label_samples = df[df['label'] == label]
            
            # Calculate samples per label
            samples_per_label_train = min(train_size // len(labels), len(label_samples))
            samples_per_label_test = min(test_size // len(labels), len(label_samples) - samples_per_label_train)
            
            # Sample randomly for train and test
            train_label_samples = label_samples.sample(n=samples_per_label_train, random_state=42)
            remaining_samples = label_samples.drop(train_label_samples.index)
            test_label_samples = remaining_samples.sample(n=samples_per_label_test, random_state=42)
            
            train_samples.append(train_label_samples)
            test_samples.append(test_label_samples)
        
        # Combine samples for this dataset
        dataset_train = pd.concat(train_samples)
        dataset_test = pd.concat(test_samples)
        
        # Add dataset name as a column
        dataset_train['source'] = dataset_name
        dataset_test['source'] = dataset_name
        
        all_train_dfs.append(dataset_train)
        all_test_dfs.append(dataset_test)

        print(f'{dataset_name} train samples: {len(dataset_train)}')
        print(f'{dataset_name} test samples: {len(dataset_test)}')
    
    # Combine all datasets
    combined_train_df = pd.concat(all_train_dfs).reset_index(drop=True)
    combined_test_df = pd.concat(all_test_dfs).reset_index(drop=True)
    
    # Shuffle the combined training data
    combined_train_df = combined_train_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Convert to Hugging Face datasets
    train_dataset = Dataset.from_pandas(combined_train_df)
    test_dataset = Dataset.from_pandas(combined_test_df)
    
    return DatasetDict({
        'train': train_dataset.rename_column('message', 'text'),
        'test': test_dataset.rename_column('message', 'text')
    })

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="distilbert-base-uncased",
                      help="Model name or path")
    parser.add_argument("--output_dir", type=str, default="./results",
                      help="Directory to save the model")
    parser.add_argument("--num_train_epochs", type=int, default=3,
                      help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16,
                      help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, default=2e-5,
                      help="Learning rate")
    parser.add_argument("--max_length", type=int, default=128,
                      help="Maximum sequence length")
    parser.add_argument("--data_dir", type=str, default="data",
                      help="Directory containing the CSV files")
    args = parser.parse_args()

    # Load datasets
    datasets_dict = load_datasets(args.data_dir)
    
    # Prepare balanced datasets
    dataset = prepare_balanced_datasets(datasets_dict)

    print(f'dataset: {dataset}')
    
    # Load tokenizer and model
    tokenizer = DistilBertTokenizer.from_pretrained(args.model_name)
    num_labels = len(set(dataset['train']['label']))
    model = DistilBertForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=num_labels
    )

    # Preprocess dataset
    tokenized_dataset = dataset.map(
        lambda x: preprocess_function(x, tokenizer, args.max_length),
        batched=True
    )

    # Define training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.num_train_epochs,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
        eval_strategy='epoch',
        logging_strategy='epoch',
        save_strategy='no'
    )

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"],
        compute_metrics=compute_metrics
    )

    # Train the model
    trainer.train()

    # Save the final model
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

if __name__ == "__main__":
    main() 