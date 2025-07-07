# metrics.py

from sklearn.metrics import accuracy_score, precision_recall_fscore_support


def compute_metrics(eval_preds):
    """
    计算准确率、精确率、召回率、F1 值。
    """
    preds, labels = eval_preds
    preds = preds.argmax(-1)

    acc = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='macro')

    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }
