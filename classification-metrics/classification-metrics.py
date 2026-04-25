import numpy as np

def cal(y_true, y_pred, target):
    tp = np.sum((y_true == y_pred) & (y_pred == target))
    fp = np.sum((y_pred == target) & (y_true != target))
    fn = np.sum((y_true == target) & (y_pred != target))
    return tp, fp, fn

def get_metrics(tp, fp, fn):
    precision = tp / (fp + tp) if (fp + tp) != 0 else 0
    recall = tp / (fn + tp) if (fn + tp) != 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0
    return precision, recall, f1

def classification_metrics(y_true, y_pred, average="micro", pos_label=1):
    """
    Compute accuracy, precision, recall, F1 for single-label classification.
    Averages: 'micro' | 'macro' | 'weighted' | 'binary' (uses pos_label).
    Return dict with float values.
    """
    # Write code here
    res = {"accuracy" : -1, "precision" : -1, "recall": -1, "f1" : -1}
    
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    # calculate accuracy 
    acc = np.sum(y_true == y_pred) / len(y_pred)
    
    k = np.max(y_true) + 1
    
    precision_k = np.zeros(k) 
    recall_k = np.zeros(k)
    f1_k = np.zeros(k)

    for c in range(k):
        tp, fp, fn = cal(y_true, y_pred, c)
        p_c, r_c, f1_c = get_metrics(tp, fp, fn)
        
        precision_k[c] = p_c
        recall_k[c] = r_c
        f1_k[c] = f1_c

    
    print("precision: ", precision_k)
    
    print("recall: ", recall_k)
        
    if average == "micro":
        
        res["precision"] = acc
        res["recall"] = acc
        res["f1"] = acc
        
    elif average == "macro":
        p_macro = np.sum(precision_k) / k
        r_macro = np.sum(recall_k) / k
        f1_macro = np.sum(f1_k) / k
        
        res["precision"] = p_macro
        res["recall"] = r_macro
        res["f1"] =  f1_macro
            
    elif average == "weighted":
        p_w = 0.
        r_w = 0.
        f1_w = 0.
        for i in range(k):
            support_i = np.sum(y_true == i)
            p_w += precision_k[i] * (support_i / len(y_true))
            r_w += recall_k[i] * (support_i / len(y_true))
            f1_w += f1_k[i] * (support_i / len(y_true))
        
        res["precision"] = p_w
        res["recall"] = r_w
        res["f1"] =  f1_w
        
    elif average == "binary":
        res["precision"] = precision_k[pos_label]
        res["recall"] = recall_k[pos_label]
        res["f1"] =  f1_k[pos_label]
    else:
        return None
    

    res["accuracy"] = acc

    return res 