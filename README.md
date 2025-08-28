# Dataset Preparation

This project reads MLS listing data from `property_v2.csv` instead of a database connection string.

## What it does
- Loads the raw CSV file
- Keeps only rows with non-null `ClosePrice`
- Splits data into train/test sets (default 80/20)
- Writes outputs to:
  - `data/train.csv`
  - `data/test.csv`

## Run
```bash
python prepare_dataset.py
```

## Optional arguments
```bash
python prepare_dataset.py --input property_v2.csv --output-dir data --test-size 0.2 --random-state 42
```
