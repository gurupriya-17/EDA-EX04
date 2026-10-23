# Extracting an Email Dataset and Performing EDA using Pandas

## Overview
This project loads an email CSV dataset, explores its structure, checks data quality,
cleans duplicate rows, computes email length and word-count features, and creates charts.

## Features
- Load email data from CSV with Pandas
- Display sample rows, shape, columns, and data types
- Check missing values and duplicate records
- Analyze email length and word count
- Visualize category distribution when a label column is present
- Save cleaned data, charts, and a summary report

## Project structure
```text
email-dataset-eda/
├── email_eda.py
├── emails.csv
├── requirements.txt
├── README.md
└── outputs/
```

## Dataset format
Place your dataset in this folder as `emails.csv`.

The script tries to identify common message columns such as `email`, `text`, `message`,
`body`, `content`, `v1`, and `v2`, and category columns such as `label`, `category`,
`class`, `spam`, `v1`, and `v2`.

For best results, use clear column names such as:
```csv
label,text
ham,"Hi, are we meeting today?"
spam,"Congratulations, you have won a prize!"
```

If your dataset uses different column names, rename them to `label` and `text`, or
update the candidate lists in `email_eda.py`.

## Run locally
```bash
python -m venv .venv
```

Windows:
```bash
.venv\Scripts\activate
```

Install dependencies:
```bash
pip install -r requirements.txt
```

Run:
```bash
python email_eda.py
```

## Output
The `outputs/` folder will contain:
- `cleaned_emails.csv`
- `eda_summary.txt`
- `email_length_distribution.png`
- `word_count_distribution.png`
- `category_distribution.png` (when a category column is detected)

## Note
Use a dataset you have permission to analyze. Avoid uploading real personal emails or
sensitive information to a public GitHub repository. This is an exploratory analysis
project, not a spam classifier.
