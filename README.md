# Phishing Detection Models

This project implements and evaluates machine learning models for detecting phishing attempts in both SMS messages and emails.

## Project Structure

```
.
├── data/
│   ├── phishing_sms_dataset_unique.csv    # Dataset for SMS phishing detection
│   └── unified_phishing_email_dataset.csv  # Dataset for email phishing detection
├── email_model_evaluation.ipynb           # Jupyter notebook for evaluating email dataset
├── inference_evaluation.py                # Script for model inference evaluation
├── sms_model_evaluation.ipynb            # Jupyter notebook for evaluating SMS dataset
└── requirements.txt                      # Python dependencies
```

## Setup

1. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Components

### SMS Phishing Detection
- Uses the `phishing_sms_dataset_unique.csv` dataset
- Model evaluation and analysis can be found in `sms_model_evaluation.ipynb`

### Email Phishing Detection
- Uses the `unified_phishing_email_dataset.csv` dataset
- Model evaluation and analysis can be found in `email_model_evaluation.ipynb`

## Requirements

See `requirements.txt` for a complete list of Python dependencies.

## Usage
1. First, ensure all dependencies are installed
2. Run the Jupyter notebooks to see the model evaluation and analysis:
   ```bash
   jupyter notebook
   ```
3. For sample inference evaluation, run:
   ```bash
   python inference_evaluation.py
   ```
