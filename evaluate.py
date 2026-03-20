import pandas as pd
import joblib
import os
import plotly.figure_factory as ff
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def evaluate():
    preds = pd.read_csv("data/test_predictions.csv")
    y_true = preds["true"]
    y_pred = preds["pred"]

    labels = ["Positive", "Negative", "Neutral"]

    print("=" * 45)
    print(f"  Accuracy: {accuracy_score(y_true, y_pred)*100:.1f}%")
    print("=" * 45)
    print(classification_report(y_true, y_pred, labels=labels))

    # ── Confusion matrix ──────────────────────────────────────────────────────
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_list = cm.tolist()

    fig = ff.create_annotated_heatmap(
        z=cm_list,
        x=labels, y=labels,
        colorscale=[[0, "#f0faf5"], [1, "#1a7a4a"]],
        showscale=True
    )
    fig.update_layout(
        title="Confusion Matrix",
        xaxis_title="Predicted",
        yaxis_title="Actual",
        font=dict(family="DM Sans", size=13),
        paper_bgcolor="white",
        plot_bgcolor="white",
        width=500, height=450
    )
    fig.update_yaxes(autorange="reversed")

    os.makedirs("data", exist_ok=True)
    fig.write_image("data/confusion_matrix.png")
    print("[evaluate] Saved → data/confusion_matrix.png")

    return accuracy_score(y_true, y_pred), cm

if __name__ == "__main__":
    evaluate()