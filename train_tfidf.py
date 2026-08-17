import os
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

BASE = os.path.dirname(__file__)


def load_csvs(folder):
    dfs = []
    for f in os.listdir(folder):
        if f.endswith(".csv"):
            df = pd.read_csv(os.path.join(folder, f), encoding="utf-8", on_bad_lines="skip")
            if "question " in df.columns:
                df = df.rename(columns={"question ": "question"})
            if "question" in df.columns and "intent" in df.columns:
                dfs.append(df[["question", "intent"]])
    return pd.concat(dfs, ignore_index=True)


def main():
    train_df = load_csvs(os.path.join(BASE, "qi_train")).dropna()
    test_df = load_csvs(os.path.join(BASE, "qi_test")).dropna()

    # 한국어는 형태소 분석기 없이 어절 토큰화가 약해서 char n-gram이 더 안전함
    vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=1)
    X_train = vectorizer.fit_transform(train_df["question"])

    clf = LogisticRegression(max_iter=2000, C=10.0)
    clf.fit(X_train, train_df["intent"])

    X_test = vectorizer.transform(test_df["question"])
    preds = clf.predict(X_test)
    acc = accuracy_score(test_df["intent"], preds)
    f1 = f1_score(test_df["intent"], preds, average="macro")
    print(f"qi_test accuracy={acc:.4f} macro_f1={f1:.4f}")

    out_dir = os.path.join(BASE, "model")
    os.makedirs(out_dir, exist_ok=True)
    joblib.dump(vectorizer, os.path.join(out_dir, "vectorizer.joblib"))
    joblib.dump(clf, os.path.join(out_dir, "clf.joblib"))
    print(f"saved to {out_dir}")


if __name__ == "__main__":
    main()
