"""
Project: Extracting an Email Dataset and Performing EDA using Pandas
Run: python email_eda.py
Input: emails.csv
Expected columns: email/text/message and optionally label/category
"""

from pathlib import Path
import re
import pandas as pd
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "emails.csv"
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)


def find_column(df, candidates):
    """Return the first matching column name, ignoring case and spaces."""
    normalized = {str(col).strip().lower(): col for col in df.columns}
    for candidate in candidates:
        if candidate in normalized:
            return normalized[candidate]
    return None


def main():
    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_FILE}\n"
            "Place your CSV dataset in this project folder and name it emails.csv."
        )

    # 1. Extract/load the dataset
    df = pd.read_csv(DATA_FILE, encoding="utf-8", on_bad_lines="skip")
    df.columns = [str(col).strip() for col in df.columns]

    print("\n========== EMAIL DATASET EDA ==========")
    print("\nFirst 5 rows:")
    print(df.head().to_string(index=False))

    print("\nDataset shape (rows, columns):", df.shape)
    print("\nColumn names:", df.columns.tolist())
    print("\nData types:")
    print(df.dtypes.to_string())

    # 2. Data quality checks
    print("\nMissing values per column:")
    print(df.isnull().sum().to_string())

    duplicate_count = int(df.duplicated().sum())
    print(f"\nDuplicate rows: {duplicate_count}")

    # 3. Identify likely email-text and label columns
    text_col = find_column(
        df, ["email", "text", "message", "body", "email_text", "content", "v2"]
    )
    label_col = find_column(
        df, ["label", "category", "class", "spam", "target", "v1"]
    )

    if text_col is None:
        # Use the first object/string column as a practical fallback.
        string_cols = df.select_dtypes(include=["object", "string"]).columns.tolist()
        if not string_cols:
            raise ValueError(
                "No text column found. Rename your email-message column to 'text'."
            )
        text_col = string_cols[0]
        print(f"\nCould not identify a standard text column; using '{text_col}'.")

    # 4. Clean the dataset
    df[text_col] = df[text_col].fillna("").astype(str)
    df = df.drop_duplicates().copy()
    df["email_length"] = df[text_col].str.len()
    df["word_count"] = df[text_col].apply(
        lambda message: len(re.findall(r"\b\w+\b", message))
    )

    print("\nText column used:", text_col)
    print("\nEmail length statistics:")
    print(df["email_length"].describe().round(2).to_string())

    print("\nWord count statistics:")
    print(df["word_count"].describe().round(2).to_string())

    # 5. Label analysis, if a label/category column exists
    if label_col is not None and label_col != text_col:
        print("\nEmail category counts:")
        print(df[label_col].fillna("Unknown").value_counts().to_string())

        category_counts = df[label_col].fillna("Unknown").value_counts()
        plt.figure(figsize=(8, 5))
        category_counts.plot(kind="bar")
        plt.title("Email Dataset: Category Distribution")
        plt.xlabel("Category")
        plt.ylabel("Number of emails")
        plt.xticks(rotation=0)
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "category_distribution.png", dpi=150)
        plt.close()
    else:
        print(
            "\nNo label/category column detected. "
            "Skipping spam/ham category analysis."
        )

    # 6. Visualize email lengths
    plt.figure(figsize=(8, 5))
    df["email_length"].plot(kind="hist", bins=30, edgecolor="black")
    plt.title("Distribution of Email Lengths")
    plt.xlabel("Number of characters")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "email_length_distribution.png", dpi=150)
    plt.close()

    # 7. Visualize word counts
    plt.figure(figsize=(8, 5))
    df["word_count"].plot(kind="hist", bins=30, edgecolor="black")
    plt.title("Distribution of Email Word Counts")
    plt.xlabel("Number of words")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "word_count_distribution.png", dpi=150)
    plt.close()

    # 8. Save cleaned data and a summary report
    cleaned_path = OUTPUT_DIR / "cleaned_emails.csv"
    df.to_csv(cleaned_path, index=False)

    summary = [
        "EMAIL DATASET EDA SUMMARY",
        "=========================",
        f"Rows after duplicate removal: {len(df)}",
        f"Number of columns after feature creation: {len(df.columns)}",
        f"Text column: {text_col}",
        f"Original duplicate rows: {duplicate_count}",
        f"Missing values in original data:\n{df.isnull().sum().to_string()}",
        f"\nEmail length statistics:\n{df['email_length'].describe().to_string()}",
        f"\nWord count statistics:\n{df['word_count'].describe().to_string()}",
    ]
    if label_col is not None and label_col != text_col:
        summary.append(
            f"\nCategory counts:\n"
            f"{df[label_col].fillna('Unknown').value_counts().to_string()}"
        )

    (OUTPUT_DIR / "eda_summary.txt").write_text(
        "\n".join(summary), encoding="utf-8"
    )

    print("\nEDA completed successfully!")
    print("Cleaned dataset:", cleaned_path)
    print("Summary report:", OUTPUT_DIR / "eda_summary.txt")
    print("Charts saved in:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
