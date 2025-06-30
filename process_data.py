import pandas as pd
import os

def process_telegram_data():
    # Read the input CSV file
    input_path = os.path.join('data', 'telegram_dataset.csv')
    output_path = os.path.join('data', 'telegram_dataset_processed.csv')
    
    # Read the CSV file
    print("Reading the dataset...")
    df = pd.read_csv(input_path)
    
    # Convert text_type to numerical values
    print("Converting text_type to numerical values...")
    df['label'] = (df['text_type'] == 'spam').astype(int)
    
    # Save the processed dataset
    print("Saving the processed dataset...")
    df.to_csv(output_path, index=False)
    print(f"Processed data saved to: {output_path}")
    
    # Print some statistics
    print("\nDataset statistics:")
    print(f"Total samples: {len(df)}")
    print(f"Spam messages (1): {df['label'].sum()}")
    print(f"Non-spam messages (0): {len(df) - df['label'].sum()}")


def process_email_data():
    # Read the input CSV file
    input_path = os.path.join('data', 'unified_phishing_email_dataset.csv')
    output_path = os.path.join('data', 'email_dataset_processed.csv')
    
    # Read the CSV file
    print("Reading the dataset...")
    df = pd.read_csv(input_path)
    # delete the NaN value from df
    df = df.dropna(subset=['subject', 'body'])
    
    # concatenate the subject and body columns
    df['message'] = df['subject'] + ' ' + df['body']

    # only keep the message and label columns
    df = df[['message', 'label']]

    # save the processed dataset
    df.to_csv(output_path, index=False)
    print(f"Processed data saved to: {output_path}")
    
    # Print some statistics
    print("\nDataset statistics:")
    print(f"Total samples: {len(df)}")
    print(f"Spam messages (1): {df['label'].sum()}")
    print(f"Non-spam messages (0): {len(df) - df['label'].sum()}")

if __name__ == "__main__":
    # process_telegram_data() 
    process_email_data()