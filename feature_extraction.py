from extractors.lbp import LocalBinaryPatterns
import numpy as np

extractors = [LocalBinaryPatterns]

def get_features(stack):
  feature_stack = np.empty(shape=(len(stack), len(extractors)))
  for i, image in enumerate(stack):
    for j, extractor in enumerate(extractors):
      feature_stack[i, j] = extractor(image)  

    return feature_stack
  
def save_features():
  pass