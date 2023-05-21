import sys
import os
# Get the parent directory path
parent_dir = os.path.abspath(os.path.join(os.getcwd(), "."))
# Add the parent directory to the Python path
sys.path.append(parent_dir)
# print(sys.path)
# Now we can import the tools module

from tools import read_images, binary_paths
from stack import read_features, split_data
from metrics import null_accuracy

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.metrics import accuracy_score, roc_auc_score, average_precision_score, mean_absolute_percentage_error, f1_score, r2_score, cohen_kappa_score, recall_score
from sklearn.metrics import make_scorer
from sklearn.model_selection import cross_val_score

import pandas as pd
from tqdm import tqdm

"""# TODO: Try with MNIST
# You can try to list parameters of classifier here.
params = {"l1_regularization": 0.1}
def eval_classifiers(train_X, train_y, test_X, test_y):
    # Pseudo -code. Do not RUN yet.
    # Example classifiers: https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html 
    clfs = []
    preds = []
    for clf in clfs:
        train_yhat = clf.fit(train_X)
        preds.append(train_yhat)
        # You can apply CV.
        test_yhat = clf.predict(test_y)

    # Use sklearn metrics AUC.
    train_performance = auc(train_y, train_yhat)
    test_performance = auc(test_y, test_yhat)
    # You can create a table with pandas.

    return  train_performance, test_performance
"""

# TODO: Try with MNIST
# You can try to list parameters of classifier here.
def eval_classifiers(train_X, train_y, test_X, test_y, **kwargs):
    """Pseudo -code. Do not RUN yet."""
    # Example classifiers: https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html 
    classifiers = [
    KNeighborsClassifier(3),
    SVC(kernel="linear", C=0.025),
    SVC(gamma=2, C=1),
    GaussianProcessClassifier(1.0 * RBF(1.0)),
    DecisionTreeClassifier(max_depth=5),
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    MLPClassifier(alpha=1, max_iter=1000),
    AdaBoostClassifier(),
    GaussianNB(),
    QuadraticDiscriminantAnalysis(),
    ]

    metrics = [accuracy_score,
               roc_auc_score,
               average_precision_score,
               mean_absolute_percentage_error,
               f1_score,
               r2_score,
               recall_score,
               cohen_kappa_score
               ]
    
    # Define the list of scoring metrics
    cv_metrics = [
        make_scorer(accuracy_score),
        make_scorer(roc_auc_score),
        make_scorer(average_precision_score),
        make_scorer(mean_absolute_percentage_error),
        make_scorer(f1_score),
        make_scorer(r2_score),
        make_scorer(recall_score),
        make_scorer(cohen_kappa_score)
    ]

    metrics_to_str = {accuracy_score: 'accuracy',
               roc_auc_score: 'roc_auc',
               average_precision_score: 'ap',
               mean_absolute_percentage_error: 'mapr',
               f1_score: 'f1',
               r2_score: 'r2'}
    
    clfs_to_str = {
    KNeighborsClassifier(3): 'knn',
    SVC(kernel="linear", C=0.025): 'svc',
    SVC(gamma=2, C=1): 'svc_gamma',
    GaussianProcessClassifier(1.0 * RBF(1.0)): 'gpr',
    DecisionTreeClassifier(max_depth=5): 'dtc',
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1): 'rf',
    MLPClassifier(alpha=1, max_iter=1000): 'mlp',
    AdaBoostClassifier(): 'ada',
    GaussianNB(): 'gnb',
    QuadraticDiscriminantAnalysis(): 'qda',
    }

    preds = {}

    train_metrics = [f"train_{metrics_to_str[metric]}" for metric in metrics ]
    test_metrics = [f"test_{metrics_to_str[metric]}" for metric in metrics ]

    columns = []

    for m in train_metrics:
        columns.append(m)
    
    for m in test_metrics:
        columns.append(m)

    classifiers_str = [str(clf) for clf in classifiers]

    df = pd.DataFrame(index=classifiers_str, columns=columns)
    
    for i, clf in tqdm(enumerate(classifiers), desc="Classifiers are running...."):
        # ax = plt.subplot(len(classifiers) + 1, i)
        clf = make_pipeline(StandardScaler(), clf)
        clf.fit(train_X, train_y)
        score = clf.score(test_X, test_y)
        
        # Apply cross-validated model here.
        cv_scores = cross_val_score(clf, train_X, train_y, cv=10, scoring=cv_metrics)  # Specify the list of scoring metrics
        cv_mean_scores = cv_scores.mean(axis=0)  # Compute the mean score for each metric
    
        train_yhat = clf.predict(train_X)
        test_yhat = clf.predict(test_X)
        preds[str(clf)] = [train_yhat, test_yhat]

        # Use sklearn metrics AUC.
        for j, metric in enumerate(metrics):
            df.loc[classifiers_str[i], f"train_{metrics_to_str[metric]}"] = metric(train_y, train_yhat)
            df.loc[classifiers_str[i], f"test_{metrics_to_str[metric]}"] = metric(test_y, test_yhat)
            df.loc[classifiers_str[i], "score"] = score
            df.loc[classifiers_str[i], f"cv_{metrics_to_str[metric]}"] = cv_mean_scores[j]

    print(df)
    
    info = kwargs["info"]
    filename = 'classifiers/results/'
    for ex in info['extractors']:
        filename += ex
    
    filename += '_'
    filename += info['mf']
    filename += info['mode']
    filename += '.csv'

    df.to_csv(filename)
    return  preds, df

if __name__ == "__main__":
    # Use here to test MNIST or other dataset.
    extractors = ['lbp', 'fos', 'glcm', 'hos', 'pftas']
    fnames, X, y = read_features(extractors, root='features/all/', mode='binary', mf='40X')

    # print(len(fnames), len(X), len(y))
        
    X_train, X_test, y_train, y_test = split_data(X, y, one_hot_vector=False, test_size=0.3)

    # print(len(X_train), len(X_test), len(y_train), len(y_test))
    # print(X_test)
    _, performance = eval_classifiers(X_train, y_train, X_test, y_test, info={'extractors': extractors,
                                                                              'mode': 'binary',
                                                                              'mf': '40X'})
    print(performance)

