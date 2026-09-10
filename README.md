# Bank Detection System

Machine Learning based credit-card transaction fraud detection project.

## Workflow
Dataset → Preprocessing → Feature Encoding → Train/Test Split → Random Forest → Prediction → Evaluation

## Dataset
Place the required dataset at:

`dataset/credit_card_fraud_10k.csv`

The dataset should contain the target column:

`is_fraud`

## Run

```bash
pip install -r requirements.txt
python bank_detection.py
```

## Output
The program generates:

- `fraud_detection_predictions.csv`
- `model_evaluation_report.csv`

Evaluation metrics:
- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix

 
