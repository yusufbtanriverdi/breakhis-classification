"""TODO : Superpixel feature inspection."""

from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float

from cv2 import ximgproc
import cv2
import numpy as np

# Reproduce https://stackoverflow.com/questions/57039504/what-is-the-difference-between-opencv-ximgproc-slic-and-skimage-segmentation-sli

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from skimage import io

    image = img_as_float(io.imread('./examples/SOB_B_A-14-22549AB-40-019.png'))

    # apply SLIC and extract (approximately) the supplied number
    # of segments
    # TODO: Improve (fine the best parameters or change the library to opencv -- the ones that teacher used.)
    segments = slic(image, n_segments = 50, sigma = 10)
    # show the output of SLIC
    fig = plt.figure("Superpixels -- %d segments" % (100))
    ax = fig.add_subplot(1, 1, 1)
    ax.imshow(mark_boundaries(image, segments, color=(255, 0, 0)))
    plt.axis("off")
    # show the plots
    plt.show()
    
    src = cv2.GaussianBlur(image,(5,5),0)
    # Convert to LAB
    src_lab = cv2.cvtColor(np.array(src, dtype=np.uint8),cv2.COLOR_BGR2LAB) # convert to LAB
    cv_slico = ximgproc.createSuperpixelSLIC(np.array(src_lab, dtype=np.uint8), algorithm=ximgproc.SLICO, region_size=32)	
    cv_slico.iterate(num_iterations=50)
    print(cv_slico.getNumberOfSuperpixels())

    plt.imshow(cv_slico.getLabelContourMask())
    plt.show()