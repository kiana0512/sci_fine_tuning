# sci_fine_tuning/metrics.py

from sklearn.metrics import accuracy_score, precision_recall_fscore_support


def compute_metrics(eval_preds):
    """
    支持 (preds, labels) 结构的评估函数。
    """
    preds, labels = eval_preds
    acc = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='macro', zero_division=0)

    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }
