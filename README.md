## Some highlights

- [1] states *"...A comprehensive set of experiments shows that accuracy rates with this baseline system range from 80% to 85%, depending on the image magnification factor. ..."*. Our primary goal is **to exceed %85** in worst case scenario. It is also stated that they use some feature extraction methods. Let's list them here and we can use them for our models.

- We may seperately look into magnification factors classification. See Table 2 in [1]

## TODO List

- List of preprocessing methods.
- List feature extraction methods listed in [1] and find approriate libraries with some examples. We may need to reproduce them.
- List machine learning baseline models that we will have to use. These are mentioned in slides as:
    - Dynamic selection of classifiers,
    - SVM
    - RF
    - Ensemble across magnification factors 
    - Rejection scheme.
- List deep learning models that we will have to use. RetinaNet requires box annotations. We do not have that. But we may still use Focal Loss which is useful in imbalanced datasets.
- Capsulate these methods in this repository.
- Then we will start processing images.
- List metric methods that we can use.
- Visualization applications.

### References

- [1] A Dataset for Breast Cancer Histopathological Image Classification, Fabio A. et. al.