# Phishing Detection Models

This project finetuned a BERT model that  and evaluates machine learning models for detecting phishing attempts in various communication channels including SMS messages, emails, and Telegram messages.

## Project Structure

```
├── model_eval_notebook/                  # Model evaluation notebooks
│   ├── email_model_evaluation.ipynb      # Email dataset evaluation
│   ├── sms_model_evaluation.ipynb        # SMS dataset evaluation
│   └── telegram_model_evaluation.ipynb   # Telegram dataset evaluation
├── inference_evaluation.py              # Script for model evaluation
├── process_data.py                      # Data preprocessing script for telegram dataset
├── train.py                             # Model Finetuning script
└── requirements.txt                     # Python dependencies
```

## Components

### Data Processing
- `process_data.py`: Preprocesses and prepares Telegram dataset for BERT model fine-tuning

### Model Fine-tuning
- `train.py`: Implements BERT model fine-tuning pipeline
  - Fine-tunes pre-trained BERT model on phishing detection task
  - Adapts BERT's architecture for multi-channel phishing detection

### Model Evaluation
Comprehensive evaluation notebooks for each communication channel:
- Email Dataset Evaluation (`email_model_evaluation.ipynb`):
  - Evaluates BERT model performance on email phishing detection
  - Analyzes different models behavior on phishing email data
  
- SMS Dataset Evaluation (`sms_model_evaluation.ipynb`):
  - Evaluates BERT model performance on sms phishing detection
  - Analyzes different models behavior on phishing sms data
  
- Telegram Dataset Evaluation (`telegram_model_evaluation.ipynb`):
  - Evaluates BERT model performance on Telegram phishing detection
  - Analyzes different models behavior on phishing telegram data


## Requirements

See `requirements.txt` for a complete list of Python dependencies.

## Usage
1. First, ensure all dependencies are installed:
   ```bash
   pip install -r requirements.txt
   ```

2. Process the data:
   ```bash
   python process_data.py
   ```

3. Fintuned the models:
   ```bash
   python finetune.py --data_dir data/finetuned_dataset
   ```

## Note
In the evaluation notebooks, the fine-tuned BERT model's results are labeled as 'results' in all comparison charts.