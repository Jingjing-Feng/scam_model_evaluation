import os
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, precision_recall_fscore_support
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
from tqdm import tqdm
from dotenv import load_dotenv


def load_sms_dataset(file_path):
    """Load and prepare the dataset."""
    df = pd.read_csv(file_path)
    return df['message'].values, df['label'].values


def load_email_dataset(file_path):
    """Load and prepare the dataset."""
    df = pd.read_csv(file_path)
    # delete the NaN value from df
    df = df.dropna(subset=['subject', 'body'])
    
    # concatenate the subject and body columns
    df['message'] = df['subject'] + ' ' + df['body']
    return df['message'].values, df['label'].values


def sample_balanced_data(texts, labels, n_samples=100):
    """Sample equal number of spam and non-spam messages."""
    # Convert to DataFrame
    df = pd.DataFrame({'text': texts, 'label': labels})
    
    # Get n_samples/2 samples from each class
    n_per_class = n_samples // 2
    
    # Sample from each class
    spam = df[df['label'] == 1].sample(n=n_per_class, random_state=42)
    non_spam = df[df['label'] == 0].sample(n=n_per_class, random_state=42)
    
    # Combine and shuffle
    balanced_df = pd.concat([spam, non_spam]).sample(frac=1, random_state=42)
    
    print(f"\nSampled balanced dataset:")
    print(f"Total samples: {len(balanced_df)}")
    print(f"Spam samples: {len(balanced_df[balanced_df['label'] == 1])}")
    print(f"Non-spam samples: {len(balanced_df[balanced_df['label'] == 0])}")
    
    return balanced_df['text'].values, balanced_df['label'].values


def chunk_text_by_tokens(text, tokenizer, token_limit=400):
    """
    Split text into chunks based on token length using the tokenizer's encoding functionality.
    We use token_limit=512 to leave some room for special tokens.
    """

    tokenize_text = tokenizer(text)["input_ids"][1:-1]  # remove [CLS] and [SEP]
    chunked_token = [
        tokenize_text[i : i + token_limit]
        for i in range(0, len(tokenize_text), token_limit - 2)
    ]  # -2 to take [CLS] and [SEP] into consideration when tokenizing
    chunked_text = [
        tokenizer.decode(chunk, skip_special_tokens=True) for chunk in chunked_token
    ]

    return chunked_text


def evaluate_model(texts, true_labels, model_name="mshenoda/roberta-spam", spam_label="LABEL_1"):
    """Evaluate the model using the Hugging Face pipeline."""
    predicted_labels = []
    
    try:
        # Load tokenizer separately to use for chunking
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Create pipeline
        classifier = pipeline("text-classification", model=model_name, tokenizer=tokenizer)
        
        # Process each text
        for text, true_label in zip(texts, true_labels):
            try:
                # Split text into chunks based on tokens
                chunks = chunk_text_by_tokens(text, tokenizer)
                # print(f"Chunked length: {len(chunks)} chunks")
                
                # Get predictions for all chunks
                chunk_predictions = []
                for chunk in chunks:
                    result = classifier(chunk)[0]
                    # Store prediction probability for spam class
                    prob = result['score'] if result['label'] == spam_label else 1 - result['score']
                    chunk_predictions.append(prob)
                
                # Average the predictions across chunks
                avg_prob = sum(chunk_predictions) / len(chunk_predictions)
                # Convert probability to binary prediction
                pred_label = 1 if avg_prob >= 0.5 else 0
                predicted_labels.append(pred_label)
                
            except Exception as e:
                print(f"Error processing text: {str(e)}")
                predicted_labels.append(-1)
            
    except Exception as e:
        print(f"Error processing batch: {e}")
        # In case of error, predict the majority class for all samples
        predicted_labels = [0] * len(texts)
    
    # Calculate accuracy metrics
    accuracy = accuracy_score(true_labels, predicted_labels)
    print("\nModel Evaluation Results:")
    print("-------------------------")
    print(f"Accuracy: {accuracy:.4f}")
    
    # Calculate additional metrics
    precision, recall, f1, _ = precision_recall_fscore_support(true_labels, predicted_labels, average='binary')
    
    print("\nDetailed Metrics:")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    
    return {
        'predicted_labels': predicted_labels,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }


def main():
    # Load the dataset
    print("Loading dataset...")
    texts, labels = load_sms_dataset("data/phishing_sms_dataset_unique.csv")
    
    # Sample balanced dataset
    texts, labels = sample_balanced_data(texts, labels, n_samples=2)
    
    print(f"Evaluating model on {len(texts)} samples...")
    results = evaluate_model(texts, labels)
    
    # You can access individual metrics from the results dictionary if needed
    print("\nSummary of Results:")
    print(f"Final Accuracy: {results['accuracy']:.4f}")
    print(f"Final F1 Score: {results['f1']:.4f}")

if __name__ == "__main__":
    main() 

