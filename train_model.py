import json
import os
import pickle
import re

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split


def clean_text(text):
    """Clean and normalize article text for model training and prediction."""
    if pd.isna(text):
        return ""

    text = str(text).lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def validate_columns(df, file_name):
    required_columns = {"title", "text"}
    missing_columns = required_columns.difference(df.columns)

    if missing_columns:
        print(
            f"Error: {file_name} is missing required column(s): "
            f"{', '.join(sorted(missing_columns))}"
        )
        return False

    return True


def main():
    fake_path = "data/Fake.csv"
    true_path = "data/True.csv"

    if not os.path.exists(fake_path):
        print("Error: data/Fake.csv is missing. Please add the dataset file and try again.")
        return

    if not os.path.exists(true_path):
        print("Error: data/True.csv is missing. Please add the dataset file and try again.")
        return

    os.makedirs("model", exist_ok=True)

    try:
        fake_df = pd.read_csv(fake_path)
        true_df = pd.read_csv(true_path)
    except Exception as error:
        print(f"Error: Could not read dataset files. Details: {error}")
        return

    if not validate_columns(fake_df, "data/Fake.csv"):
        return

    if not validate_columns(true_df, "data/True.csv"):
        return

    fake_df = fake_df.copy()
    true_df = true_df.copy()

    fake_df["label"] = 0
    true_df["label"] = 1

    data = pd.concat([fake_df, true_df], ignore_index=True)
    data["content"] = data["title"].fillna("") + " " + data["text"].fillna("")
    data["content"] = data["content"].apply(clean_text)
    data = data[data["content"] != ""]

    if data.empty:
        print("Error: No usable text remains after cleaning the dataset.")
        return

    class_counts = data["label"].value_counts()
    if len(data) < 10 or class_counts.min() < 2:
        print(
            "Error: The dataset is too small for a reliable stratified train/test split. "
            "Please add more fake and real news articles."
        )
        return

    X = data["content"]
    y = data["label"]

    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42,
            stratify=y,
        )
    except ValueError as error:
        print(f"Error: Could not split the dataset. Details: {error}")
        return

    vectorizer = TfidfVectorizer(stop_words="english", max_df=0.7, min_df=1)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_tfidf, y_train)

    y_pred = model.predict(X_test_tfidf)

    accuracy = accuracy_score(y_test, y_pred)
    report_text = classification_report(
        y_test,
        y_pred,
        labels=[0, 1],
        target_names=["Fake", "Real"],
        zero_division=0,
    )
    report_dict = classification_report(
        y_test,
        y_pred,
        labels=[0, 1],
        target_names=["Fake", "Real"],
        output_dict=True,
        zero_division=0,
    )
    matrix = confusion_matrix(y_test, y_pred, labels=[0, 1])

    print(f"Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(report_text)
    print("Confusion Matrix:")
    print(matrix)

    with open("model/fake_news_model.pkl", "wb") as model_file:
        pickle.dump(model, model_file)

    with open("model/vectorizer.pkl", "wb") as vectorizer_file:
        pickle.dump(vectorizer, vectorizer_file)

    evaluation = {
        "accuracy": accuracy,
        "classification_report": report_dict,
        "confusion_matrix": matrix.tolist(),
    }

    with open("model/evaluation.json", "w", encoding="utf-8") as evaluation_file:
        json.dump(evaluation, evaluation_file, indent=4)

    print("\nTraining complete.")
    print("Saved files:")
    print("- model/fake_news_model.pkl")
    print("- model/vectorizer.pkl")
    print("- model/evaluation.json")


if __name__ == "__main__":
    main()
