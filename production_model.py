import os
import sys
import logging
import pickle
import string
import re
import argparse

import pandas as pd
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Ensure NLTK stopwords are available
nltk.download('stopwords')

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def preprocess_text(text):
    """Lowercase, remove punctuation, stopwords, and non-alphabetic chars."""
    try:
        text = text.lower()
        text = text.translate(str.maketrans("", "", string.punctuation))
        stop_words = set(stopwords.words("english"))
        text = " ".join([word for word in text.split() if word not in stop_words])
        text = re.sub(r"[^a-zA-Z\s]", "", text)
        return text
    except Exception as e:
        logger.error(f"Error preprocessing text: {e}")
        return text


def prepare_dataset(df, text_col, target_col, mapping):
    """Prepare dataset: select cols, encode labels, preprocess text."""
    if text_col not in df.columns or target_col not in df.columns:
        logger.error(f"Required columns '{text_col}' or '{target_col}' are missing.")
        sys.exit(1)

    dfx = df[[text_col, target_col]].copy()
    dfx[target_col] = dfx[target_col].map(mapping)
    dfx["cleaned_text"] = dfx[text_col].apply(preprocess_text)
    dfx.dropna(inplace=True)
    return dfx


def vectorize_text(text_series):
    """Convert text to TF-IDF vectors."""
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(text_series)
    return X, vectorizer


def train_and_evaluate(X, y, model, model_name):
    """Train model, evaluate, and return metrics + fitted model."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    metrics_dict = {
        "Model": model_name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, average="weighted", zero_division=0),
        "Recall": recall_score(y_test, y_pred, average="weighted", zero_division=0),
        "F1 Score": f1_score(y_test, y_pred, average="weighted", zero_division=0),
    }
    return metrics_dict, model


def save_model(model, vectorizer, model_dir, model_name, vec_name="vectorizer_tfidf.pkl"):
    """Save trained model + vectorizer."""
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    model_path = os.path.join(model_dir, model_name)
    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    vectorizer_path = os.path.join(model_dir, vec_name)
    if not os.path.exists(vectorizer_path):  # save vectorizer only once
        with open(vectorizer_path, "wb") as f:
            pickle.dump(vectorizer, f)

    logger.info(f"Saved {model_name} and vectorizer")


def main(args):
    # Load data
    df = pd.read_csv(args.data_path, encoding="ISO-8859-1").dropna()
    logger.info(f"Data loaded with shape {df.shape}")

    results = []

    # =========================
    # Problem 1: DI_Flag
    # =========================
    df1 = prepare_dataset(df, "ObsFullDescription", "DI_flag", {"No": 0, "Yes": 1})
    X1, vectorizer = vectorize_text(df1["cleaned_text"])
    metrics1, model1 = train_and_evaluate(X1, df1["DI_flag"], LogisticRegression(max_iter=1000), "Logistic Regression (DI_Flag)")
    results.append(metrics1)
    save_model(model1, vectorizer, args.model_dir, "problem statement 1_logistic_regression_model.pkl")

    # =========================
    # Problem 2: REPEAT_OBSERVATION
    # =========================
    df2 = prepare_dataset(df, "ObsFullDescription", "REPEAT_OBSERVATION", {"No": 0, "Yes": 1})
    X2, _ = vectorize_text(df2["cleaned_text"])
    metrics2, model2 = train_and_evaluate(X2, df2["REPEAT_OBSERVATION"], RandomForestClassifier(), "Random Forest (REPEAT_OBSERVATION)")
    results.append(metrics2)
    save_model(model2, vectorizer, args.model_dir, "problem statement 2_random_forest_model.pkl")

    # =========================
    # Problem 3: Outcome
    # =========================
    outcome_map = {
        "Not Applicable": 0,
        "Unsatisfactory": 1,
        "Needs Improvement": 2,
        "Good": 3,
        "Satisfactory": 4,
    }
    df3 = prepare_dataset(df, "ObsFullDescription", "outcome", outcome_map)
    X3, _ = vectorize_text(df3["cleaned_text"])
    metrics3, model3 = train_and_evaluate(X3, df3["outcome"], LogisticRegression(max_iter=1000), "Logistic Regression (Outcome)")
    results.append(metrics3)
    save_model(model3, vectorizer, args.model_dir, "problem statement 3_logistic_regression_model.pkl")

    # =========================
    # Problem 4: ObservationRating
    # =========================
    rating_map = {"Recommendation": 0, "Minor": 1, "Major": 2, "Critical": 3}
    df4 = prepare_dataset(df, "ObsFullDescription", "ObservationRating", rating_map)
    X4, _ = vectorize_text(df4["cleaned_text"])
    metrics4, model4 = train_and_evaluate(X4, df4["ObservationRating"], SVC(), "SVM (ObservationRating)")
    results.append(metrics4)
    save_model(model4, vectorizer, args.model_dir, "problem statement 4_support_vector_machine_model.pkl")

    # =========================
    # Final Report
    # =========================
    results_df = pd.DataFrame(results)
    print("\nModel Evaluation Results:\n", results_df.to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Production-ready multi-problem audit model training.")
    parser.add_argument("--data_path", type=str, default="AQWA_BASE_Data.csv", help="Path to the input CSV data file.")
    parser.add_argument("--model_dir", type=str, default="models", help="Directory to save trained models.")
    args = parser.parse_args()
    main(args)
