# TODO: Pattern Recognition Rate calculation

# TODO: AUC / Accuracy / F1 Score etc.
import numpy as np

from scipy import stats
 


def null_accuracy(acc, y):
    classes, len_classes = np.unique(y, return_counts=True)

    null_acc = np.max(len_classes) / len(y)
    
    # t_value, p_value = stats.ttest_ind(null_acc, acc, equal_var=False)

    return null_acc

##PSEUDO CODE
def patient_recognition_rate(img_names, y, y_pred):
    
    patients = {"ID1": imgs_list_of_indices}
    ... # Calculate the number of patients
    
    for patient, imgs_list in patients:
        patient_score = 0
        for i in imgs_list:
            if y[i] == y_pred[i]:
                nrec +=  1
            if y[i] == 1:
                np += 1

    return sum(patient_score)/len(patients)