import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix

def classification_metrics(pred_out, y_true_oh):
    """
    Function to aclculate some general metric like AUC and Accuracy.

    Args:
        pred_out (dictionary): output of predict_ontram() when applied to test sample X
        y_true_oh (numpy array): true classes for the test sample X
    """
    y_true = np.argmax(y_true_oh, axis=1)
    y_probs = pred_out['prob']
    y_pred = pred_out['class']
    y_pdf = pred_out['pdf']
    
    # accuracy
    acc = accuracy_score(y_pred, y_true)

    # auc
    if y_true_oh.shape[1] == 2:
        # auc
        auc = roc_auc_score(y_true, y_probs)
        # sensitivity
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        sens = tp / (tp + fn)
        spec = tn / (tn + fp)
        return {'accuracy': np.round(acc, decimals=4), 'sensitivity': np.round(sens, decimals=4), 'specificity': np.round(spec, decimals=4), 'auc': np.round(auc, decimals=4)}
    else:
        # auc = roc_auc_score(y_true_oh, y_pdf, multi_class="ovr")
        return {'accuracy': np.round(acc, decimals=4)} #, 'auc': np.round(auc, decimals=4)}
