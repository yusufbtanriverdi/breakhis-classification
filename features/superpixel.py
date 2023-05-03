"""TODO : Superpixel feature inspection."""

from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float

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