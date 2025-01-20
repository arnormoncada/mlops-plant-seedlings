from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


def compute_metrics(targets, preds):
    acc = accuracy_score(targets, preds.argmax(dim=1))
    prc = precision_score(targets, preds.argmax(dim=1), average="weighted")
    rcl = recall_score(targets, preds.argmax(dim=1), average="weighted")
    f1 = f1_score(targets, preds.argmax(dim=1), average="weighted")

    return {"accuracy": acc, "precision": prc, "recall": rcl, "f1": f1}
