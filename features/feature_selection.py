import os
import sys

# Get the parent directory path
parent_dir = os.path.abspath(os.path.join(os.getcwd(), "."))
# Add the parent directory to the Python path
sys.path.append(parent_dir)
# print(sys.path)
# Now we can import the tools module
from classifiers.stack import read_features

# TODO: Chi-square _X

# 1) Do we need feature selection? You will look at the correlation number between features and image labels - e.g. Fisher's or Chi-square

# 2) If so, what method is the best practically, initutiavely?
extractors = ['lbp']

img_info, features, label = read_features(extractors, root='./features/all/', mode='binary', mf='40X')

#Translation to the code
# 1) Given 'features', produce correlation matrix, scores, by applying Chi-square and/or Fisher's exact test. If the probability of correlation with the 'label' (i.e. p-value) is significant
# for all features, that means we  don't have to apply feature reduction.

# 2) If we need to apply, i.e. discard unrelated features, write a pipeline for the test- and filtering out the features.< >