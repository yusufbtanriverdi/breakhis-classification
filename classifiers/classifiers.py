import sys
import os
# Get the parent directory path
parent_dir = os.path.abspath(os.path.join(os.getcwd(), "."))
# Add the parent directory to the Python path
sys.path.append(parent_dir)
# print(sys.path)
# Now we can import the tools module

from tools import read_images, binary_paths
from .stack import prepare_data, read_data, read_features
from  .metrics import auc

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.datasets import make_moons, make_circles, make_classification
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
from sklearn.metrics import accuracy_score, roc_auc_score, average_precision_score, mean_absolute_percentage_error, top_k_accuracy_score, f1_score, r2_score

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
def eval_classifiers(train_X, train_y, test_X, test_y):
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
               top_k_accuracy_score,
               f1_score,
               r2_score,
               ]
    
    preds = {}
    performance = {}
    for i, clf in enumerate(classifiers):
        # ax = plt.subplot(len(classifiers) + 1, i)
        clf = make_pipeline(StandardScaler(), clf)
        clf.fit(train_X, train_y)
        score = clf.score(test_X, test_y)
        # DecisionBoundaryDisplay.from_estimator(
        #     clf, train_X, cmap='turbo', alpha=0.8, ax=ax, eps=0.5
        # )
        print(score)
        train_yhat = clf.predict(train_X)
        test_yhat = clf.predict(test_X)
        preds[str(clf)] = [train_yhat, test_yhat]

        m_dict = {}
        # Use sklearn metrics AUC.
        for metric in metrics:
            m_dict[str(metric)] = {"train": metric(train_y, train_yhat), 
                                   "test": metric(test_y, test_yhat)}
        performance[str(clf)] = m_dict
    # You can create a table with pandas.

    return  preds, performance

if __name__ == "__main__":
    # Use here to test MNIST or other dataset.
    extractors = ['lbp']
    fnames, X, y = read_features(extractors, root='./features/all/', mode='binary', mf='40X')
    pass