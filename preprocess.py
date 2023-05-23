import numpy as np

class Normalize(object):
    """Apply brightness normalization to the dataset. """

    def __init__(self):
        pass

    def __call__(self, sample, *args, **kwds):
        
        imgs, targets = sample

        mean = np.mean(imgs, axis = 0)
        imgs = np.array([img-mean for img in imgs])

        return imgs, targets

