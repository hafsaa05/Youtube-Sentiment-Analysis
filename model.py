import pandas as pd
import joblib
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings("ignore")

def train():
    df = pd.read_csv("data/cleaned_labeled.csv")
    df = df.dropna(subset=["processed", "label"])
    df = df[df["processed"].str.strip().str.len() > 0]

    X = df["processed"]
    y = df["label"]

    print(f"[model] Training on {len(df)} samples", flush=True)
    print(f"[model] Label distribution:\n{y.value_counts()}", flush=True)

    tfidf = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(2, 4),
        max_features=10000,
        min_df=1
    )
    X_vec = tfidf.fit_transform(X)
    print("[model] TF-IDF done", flush=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X_vec, y, test_size=0.2, random_state=42, stratify=y
    )

    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_train, y_train)
    lr_acc = accuracy_score(y_test, lr.predict(X_test))
    print(f"[model] Logistic Regression accuracy: {lr_acc*100:.1f}%", flush=True)

    svm = LinearSVC(max_iter=500, dual=False, random_state=42)
    svm.fit(X_train, y_train)
    svm_acc = accuracy_score(y_test, svm.predict(X_test))
    print(f"[model] SVM accuracy: {svm_acc*100:.1f}%", flush=True)

    best_model = lr if lr_acc >= svm_acc else svm
    best_name  = "Logistic Regression" if lr_acc >= svm_acc else "SVM"
    print(f"[model] Saving best model: {best_name}", flush=True)

    os.makedirs("models", exist_ok=True)
    joblib.dump(tfidf,      "models/tfidf.pkl")
    joblib.dump(best_model, "models/lr_model.pkl")

    y_pred = best_model.predict(X_test)
    pd.DataFrame({"true": y_test.values, "pred": y_pred}).to_csv(
        "data/test_predictions.csv", index=False
    )
    print("[model] Saved → models/tfidf.pkl, models/lr_model.pkl", flush=True)
    print("[model] Done.", flush=True)


if __name__ == "__main__":
    train()