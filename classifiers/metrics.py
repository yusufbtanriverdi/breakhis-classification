# TODO: Pattern Recognition Rate calculation

# TODO: AUC / Accuracy / F1 Score etc.
import numpy as np

from scipy import stats
 


def null_accuracy(acc, y):
    classes, len_classes = np.unique(y, return_counts=True)

    null_acc = np.max(len_classes) / len(y)
    
    # t_value, p_value = stats.ttest_ind(null_acc, acc, equal_var=False)

    return null_acc