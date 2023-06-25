import sys
import os
# Get the parent directory path
parent_dir = os.path.abspath(os.path.join(os.getcwd(), "."))
# Add the parent directory to the Python path
sys.path.append(parent_dir)
# print(sys.path)
# Now we can import the tools module
import numpy as np
from stack import read_features, split_data, read_data

from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, average_precision_score, f1_score, cohen_kappa_score, recall_score, log_loss
from sklearn.metrics import make_scorer
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_validate
from sklearn.kernel_ridge import KernelRidge
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import OneClassSVM
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import RadiusNeighborsClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.linear_model import PassiveAggressiveClassifier    
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression, Lars, ElasticNet, RidgeClassifier, BayesianRidge, Lasso
from sklearn.naive_bayes import MultinomialNB  
from sklearn.neighbors import NearestCentroid
from sklearn.svm import NuSVC
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.svm import SVC, NuSVC, LinearSVC
from sklearn.mixture import GaussianMixture

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
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
                BernoulliNB(),
                # OneClassSVM(),
                SGDClassifier(),
                RidgeClassifier(),
                PassiveAggressiveClassifier(),
                # GradientBoostingClassifier(),
                # RadiusNeighborsClassifier(),
                # Lasso(),
                # LinearSVC(),
                LogisticRegression(),
                # ElasticNet(),
                # BayesianRidge(),
                NearestCentroid(),
                # KernelRidge(alpha = 0.1),
                # NuSVC(),
                ExtraTreesClassifier(n_estimators=100, random_state=0),
                # GaussianProcessClassifier(kernel=1.0 * RBF(1.0), random_state=0),
                KNeighborsClassifier(1),
                # SVC(kernel="linear", C=1),
                # SVC(gamma='auto', C=1),   
                DecisionTreeClassifier(max_depth=20),
                RandomForestClassifier(max_depth=20, n_estimators=10, max_features=1),
                MLPClassifier(alpha=1, max_iter=1000),
                AdaBoostClassifier(),
                GaussianNB(),
                QuadraticDiscriminantAnalysis(),
                LinearDiscriminantAnalysis(),
                # GaussianMixture()
            ]

# Define the threshold for binary classification
threshold = 0.5

# Define a custom scoring function
def custom_score(y_true, y_pred, fn=accuracy_score):
    # Apply the threshold to obtain binary predictions
    y_pred_binary = np.where(y_pred >= threshold, 1, 0)

    # Calculate and return the custom metric
    # Replace this with your own custom metric calculation
    return fn(y_true, y_pred_binary)


cv_metrics = {
    'accuracy_score': make_scorer(accuracy_score),
    'cross_entropy_loss': make_scorer(log_loss),
    'average_precision_score' : make_scorer(average_precision_score, average='weighted'),
    'cohen_kappa_score' : make_scorer(cohen_kappa_score),
    'f1_score' : make_scorer(f1_score, average='weighted'),
    # 'precision_score' : make_scorer(precision_score, average='weighted'),
    'recall_score' : make_scorer(recall_score, average='weighted'),
    'roc_auc_score': make_scorer(roc_auc_score, average='weighted'),
    'specificity_score' : make_scorer(recall_score, pos_label=0, average='binary'),
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
        cv = StratifiedKFold(n_splits=10)  # Specify the number of desired folds
        cv_scores = cross_validate(clf, X, y, cv=cv, scoring=cv_metrics, return_train_score=True)  # Specify the list of scoring metrics
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
    extractors = ['glcm', 'hos','lbp', 'lpq', 'orb', 'wpd', 'resnet18', 'googlenet']

    # Stack is throuple of image, multiclass/binary label, filename.
    stack = read_data('../BreaKHis_v1/', '40X', mode = 'multiclass', shuffle=True, imsize=None)

    fnames, X, y_binary = read_features(extractors, root='features/all/', mf='40X')

    fnames = list(fnames)

    # Get the indices where fnames == stack[:, -1]
    indices = np.where(np.isin(stack[:, -1], fnames))

    # Filter indices based on y_binary == 0
    filtered_indices = indices[0][y_binary[indices[0]] == 0]

    X_negative = X[y_binary[indices[0]] == 0]
    # Rearrange stack based on the filtered indices
    rearranged_stack = stack[filtered_indices]

    # Assign stack[:, 1] as y_multiclass
    y_negative_multiclass = rearranged_stack[:, 1]

    # print(y_negative_multiclass)
    # y_multiclass = [np.argmax(y) for y in y_multiclass]
    y_multiclass_ohv = [np.array(y) for y in y_negative_multiclass]
    y_multiclass_num = [np.argmax(y) for y in y_negative_multiclass]

    # X_train, X_test, y_train, y_test = split_data(X, y_multiclass, one_hot_vector=False, test_size=0.3)
    performance = eval_classifiers(X, y_binary, info={'extractors': extractors,'mode': 'binary', 'mf': '40X'})
    # print(performance)

