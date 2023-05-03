## Installation

- Install git for your OS.

- Run `git clone https://github.com/yusuftengriverdi/breast_histopathology_clf.git`

- Create a new virtual environment with latest python, activate it and run `pip install -r requirements.txt` in terminal.

- You are good to go.

## Done

- Stacking images (from one magnification factor) as torch dataset and training them with our current deep learning is doable. However, I need extra memory:

`Allocator: not enough memory: you tried to allocate 82182144 bytes.`

- Feature extraction methods are almost done!!! For cnn models, we need to decide and define layer indices for each model. 

- I start to inspect superpixel features as it would be lighter than fixel features. 

- Pytorch built-in models are ready to use but we have to implement normalization process unique to our data.


## Some highlights

- [1] states *"...A comprehensive set of experiments shows that accuracy rates with this baseline system range from 80% to 85%, depending on the image magnification factor. ..."*. Our primary goal is **to exceed %85** in worst case scenario. It is also stated that they use some feature extraction methods. Let's list them here and we can use them for our models.

- We may seperately look into magnification factors classification. See Table 2 in [1]

- We can try with lighter DL models for prior results or train with smaller images.


## TODO List

- See issues for details. 
- List of preprocessing methods.
- ~~List feature extraction methods listed in [1] and find approriate libraries with some examples. We may need to reproduce them.~~
- List machine learning baseline models that we will have to use. These are mentioned in slides and in issues.
- ~~List deep learning models that we will have to use. RetinaNet requires box annotations. We do not have that.But we may still use Focal Loss or Pyramid Net which is useful in imbalanced datasets.~~ 
- Then we will start processing images (train, test, eval, feature extraction, etc.)
- List metric methods that we can use.
- Visualization applications.

### References

- [1] A Dataset for Breast Cancer Histopathological Image Classification, Fabio A. et. al.
