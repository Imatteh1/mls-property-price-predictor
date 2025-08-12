# Dataset Preparation

This project reads MLS listing data from `property_v2.csv` instead of a database connection string.

## What it does
- Loads the raw CSV file
- Keeps only rows with non-null `ClosePrice`
- Splits data into train/test sets with an 80/20 ratio
- Writes outputs to:
  - `data/train.csv`
  - `data/test.csv`

## Run
```bash
python prepare_dataset.py
```
