#importing required libraries
from skimage.io import imread
from skimage.transform import resize
from skimage.feature import hog
import matplotlib.pyplot as plt
import numpy as np

class HOG:
    """
    Computes h. oriented gradients of image and extracts.
    """
    def __init__(self, orientations = 9, pixels_per_cell = (8, 8), cells_per_block = (2, 2)):
        self.orientations = orientations
        self.pixels_per_cell = pixels_per_cell
        self.cells_per_block = cells_per_block


    def describe(self, image):
        # Check if the image needs to be rescaled
        # resizing image
        resized_img = resize(image, (128*4, 64*4))

        # creating hog features
        fd, hog_image = hog(resized_img, orientations=self.orientations, pixels_per_cell=self.pixels_per_cell,
                            cells_per_block=self.cells_per_block, visualize=True, channel_axis=-1)

        # Returns image, should it?
        return np.array(hog_image)

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # reading the image
    img = imread('./examples/SOB_B_A-14-22549AB-40-019.png')
    plt.axis("off")
    plt.imshow(img)
    print(img.shape)

    # resizing image
    resized_img = resize(img, (128*4, 64*4))
    plt.axis("off")
    plt.imshow(resized_img)
    print(resized_img.shape)

    # creating hog features
    fd, hog_image = hog(resized_img, orientations=9, pixels_per_cell=(8, 8),
                        cells_per_block=(2, 2), visualize=True, channel_axis=-1)
    plt.axis("off")
    plt.imshow(hog_image, cmap="gray")
    plt.show()