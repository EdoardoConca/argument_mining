from sklearn.metrics import classification_report, f1_score,precision_score, recall_score, precision_recall_fscore_support, accuracy_score

def acc_and_f1(preds, labels, target_names=None):
    """
    Compute accuracy and F1 score.
    :param preds: predicted labels
    :param labels: true labels
    :param target_names: list of class names

    :return: dictionary with accuracy, F1 score, precision, recall, and classification report
    """
    acc = accuracy_score(preds, labels)
    f1_weighted = f1_score(y_true=labels, y_pred=preds, average='weighted')
    f1_micro = f1_score(labels, preds, average='micro')
    f1_macro = f1_score(labels, preds, average='macro')
    f1_claim = f1_score(labels, preds, labels=[0, 2], average='micro')  # claim labels
    f1_evidence = f1_score(labels, preds, labels=[1, 3], average='micro')  # evidence labels
    precision = precision_score(labels, preds, average='macro')
    recall = recall_score(labels, preds, average='macro')
    clf_report = classification_report(preds, labels, target_names=target_names,zero_division=0)

    return {
        "acc": acc,
        "eval_f1_weighted": f1_weighted,
        "eval_f1_micro": f1_micro,
        "eval_f1_macro": f1_macro,
        "f1_claim": f1_claim,
        "f1_evidence": f1_evidence,
        "precision": precision,
        "recall": recall,
        "clf_report": clf_report
    }

def f1_scores(y_pred, y_true, labelfilter=None):
    """
    Compute accuracy, F1 score, precision, recall, and classification report.
    :param y_pred: predicted labels
    :param y_true: true labels
    :param labelfilter: list of labels to consider

    :return: dictionary with accuracy, F1 score, precision, recall, and classification report
    """
    acc = accuracy_score(y_pred, y_true)
    f1_micro_filtered = f1_score(y_true, y_pred, labels=labelfilter, average='micro')
    f1_macro_filtered = f1_score(y_true, y_pred, labels=labelfilter, average='macro')
    f1_micro = f1_score(y_true, y_pred, average='micro')
    f1_macro = f1_score(y_true, y_pred, average='macro')
    f1_weighted = f1_score(y_true, y_pred, average='weighted')
    precision = precision_score(y_true, y_pred, average='macro')
    recall = recall_score(y_true, y_pred, average='macro')
    clf_report = classification_report(y_true, y_pred, zero_division=0)

    return {
        "acc": acc,
        "eval_f1_micro_filtered": f1_micro_filtered,
        "eval_f1_macro_filtered": f1_macro_filtered,
        "eval_f1_micro": f1_micro,
        "eval_f1_macro": f1_macro,
        "eval_f1_weighted": f1_weighted,
        "precision": precision,
        "recall": recall,
        "clf_report": clf_report
    }

def compute_metrics(task_name, y_pred, y_true):
    assert len(y_pred) == len(y_true)
    if task_name == "seqtag":
        return acc_and_f1(y_pred, y_true)
    elif task_name == "rel_class":
        return f1_scores(y_pred, y_true, labelfilter=[0, 1])  # 0: NoRelation, 1: Relation
    else:
        raise KeyError(f"Task {task_name} not supported.")
