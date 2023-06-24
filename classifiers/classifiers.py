import sys
import os
# Get the parent directory path
parent_dir = os.path.abspath(os.path.join(os.getcwd(), "."))
# Add the parent directory to the Python path
sys.path.append(parent_dir)
# print(sys.path)
# Now we can import the tools module

from stack import read_features, split_data, read_data

import numpy as np
from sklearn.kernel_ridge import KernelRidge
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, ExtraTreesClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis, LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score, roc_auc_score, average_precision_score, f1_score, cohen_kappa_score, recall_score, log_loss
from sklearn.metrics import make_scorer
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_validate

import pandas as pd
from tqdm import tqdm
import re

def extract_text_between_parentheses(string):
    pattern = r'\((.*?)\)'  # Regular expression pattern to match text between parentheses
    match = re.search(pattern, string)  # Search for the pattern in the string

    if match:
        return match.group(1)  # Extract the text between parentheses
    else:
        return None  # Return None if no match is found


classifiers = [
KernelRidge(alpha = 0.1),
ExtraTreesClassifier(n_estimators=100, random_state=0),
GaussianProcessClassifier(kernel=1.0 * RBF(1.0), random_state=0),
KNeighborsClassifier(1),
SVC(kernel="linear", C=1),
SVC(gamma='auto', C=1),   
DecisionTreeClassifier(max_depth=20),
RandomForestClassifier(max_depth=20, n_estimators=10, max_features=1),
MLPClassifier(alpha=1, max_iter=1000),
AdaBoostClassifier(),
GaussianNB(),
QuadraticDiscriminantAnalysis(),
LinearDiscriminantAnalysis()
]


cv_metrics = {
    'accuracy_score': make_scorer(accuracy_score),
    'cross_entropy_loss': make_scorer(log_loss),
    'average_precision_score' : make_scorer(average_precision_score),
    'cohen_kappa_score' : make_scorer(cohen_kappa_score),
    'f1_score' : make_scorer(f1_score),
    'recall_score' : make_scorer(recall_score),
    'roc_auc_score': make_scorer(roc_auc_score),
    'specificity_score' : make_scorer(recall_score, pos_label=0),
}

# TODO: Try with MNIST
# You can try to list parameters of classifier here.
def eval_classifiers(X, y, **kwargs):
    """Pseudo -code. Do not RUN yet."""
    # Example classifiers: https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html 
    # Define the list of scoring metrics
    df = pd.DataFrame()
    
    for i, clf in tqdm(enumerate(classifiers), desc="Classifiers are running...."):
        # ax = plt.subplot(len(classifiers) + 1, i)
        clf_key = str(clf)
        clf = make_pipeline(StandardScaler(), clf)
        # Apply cross-validated model here.
        cv_scores = cross_validate(clf, X, y, cv=10, scoring=cv_metrics, return_train_score=True)  # Specify the list of scoring metrics
        # print(cv_scores)
        # print(np.array(cv_scores.values()))

        # Use sklearn metrics AUC.
        for j, key in enumerate(cv_scores.keys()):
            df.loc[clf_key, key] = np.mean(cv_scores[key])

        print(df)
    
    info = kwargs["info"]
    filename = f'classifiers/results/{info["mode"]}/{info["mf"]}/'
    for ex in info['extractors']:
        filename += ex
    
    filename += '_'
    filename += info['mf']
    filename += info['mode']
    filename += '.csv'

    df.to_csv(filename)
    return  df

if __name__ == "__main__":
    # Use here to test MNIST or other dataset.
    # FOS HAS MISSING VALUES!!!
    extractors = ['glcm', 'hos','lbp', 'lpq', 'orb', 'wpd']

    # Stack is throuple of image, multiclass/binary label, filename.
    stack = read_data('../BreaKHis_v1/', '40X', mode = 'multiclass', shuffle=False, imsize=None)

    fnames, X, y_binary = read_features(extractors, root='features/all/', mf='40X')

    fnames = list(fnames)

    # Get the indices where fnames == stack[:, -1]
    indices = np.where(np.isin(stack[:, -1], fnames))

    # Rearrange stack based on the indices
    rearranged_stack = stack[indices]

    # Assign stack[:, 1] as y_multiclass
    y_multiclass = rearranged_stack[:, 1]
        
    X_train, X_test, y_train, y_test = split_data(X, y_multiclass, one_hot_vector=False, test_size=0.3)
    
    # print(X_test)
    performance = eval_classifiers(X, y_binary, info={'extractors': extractors,'mode': 'multiclass', 'mf': '40X'})
    print(performance)

