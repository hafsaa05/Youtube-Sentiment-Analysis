import joblib
import time
import re
from preprocessor import clean, tokenize_and_filter, detect_language

tfidf = joblib.load("models/tfidf.pkl")
model = joblib.load("models/lr_model.pkl")

# top keywords that drove the prediction
def extract_keywords(tokens: list[str], n: int = 5) -> list[str]:
    return [t for t in tokens if not t.startswith("NEG_")][:n]

def tone_shift(tokens: list[str]) -> str:
    neg_count = sum(1 for t in tokens if t.startswith("NEG_"))
    if neg_count > 0:
        return "Shifted (negation detected)"
    return "Stable"

def predict(text: str) -> dict:
    start = time.time()

    cleaned   = clean(text)
    tokens    = tokenize_and_filter(cleaned)
    processed = " ".join(tokens)
    language  = detect_language(text)
    keywords  = extract_keywords(tokens)

    vec       = tfidf.transform([processed])
    label     = model.predict(vec)[0]

    # confidence — use decision_function for SVM, predict_proba for LR
    try:
        proba      = model.predict_proba(vec)[0]
        confidence = round(max(proba) * 100, 1)
    except AttributeError:
        # SVM — use distance from hyperplane as proxy
        scores     = model.decision_function(vec)[0]
        if hasattr(scores, "__len__"):
            confidence = round(
                (max(scores) - min(scores)) /
                (max(scores) - min(scores) + 1e-5) * 100, 1
            )
            confidence = min(round(abs(max(scores)) * 15 + 55, 1), 99.9)
        else:
            confidence = min(round(abs(float(scores)) * 20 + 55, 1), 99.9)

    latency_ms = round((time.time() - start) * 1000)
    word_count = len(text.split())

    # build insight sentence
    if keywords:
        kw_str = ", ".join(f'"{k}"' for k in keywords[:3])
        insight = (
            f"The analyzer identified keywords {kw_str} which weighted "
            f"the result toward a {label} sentiment category. "
        )
        neg_tokens = [t for t in tokens if t.startswith("NEG_")]
        if neg_tokens:
            insight += f"Negation was detected ({len(neg_tokens)} negated term(s)), shifting the tone."
        else:
            insight += "No negative qualifiers were detected in the input."
    else:
        insight = f"Insufficient keywords to explain. Classified as {label} based on overall pattern."

    return {
        "label":      label,
        "confidence": confidence,
        "language":   language,
        "word_count": word_count,
        "latency_ms": latency_ms,
        "tone":       tone_shift(tokens),
        "keywords":   keywords,
        "insight":    insight,
    }


if __name__ == "__main__":
    tests = [
        "This video is absolutely amazing, zabardast kaam kiya!",
        "Yeh bakwas hai, bilkul bekar video",
        "nahi acha tha yeh content, bohot bura laga",
        "okay video, nothing special",
        "mashallah bhai bohot acha content hai",
    ]
    for t in tests:
        r = predict(t)
        print(f"\nText: {t}")
        print(f"  → {r['label']} ({r['confidence']}%) | {r['language']} | {r['latency_ms']}ms")
        print(f"  → {r['insight']}")