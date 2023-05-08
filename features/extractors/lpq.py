from __future__ import division

import numpy as np
from scipy.signal import convolve2d
import cv2
import matplotlib.pyplot as plt

class LPQ:
    def __init__(self, winSize=3, freqestim=1, mode='nh'):
        self.winSize = winSize
        self.freqestim = freqestim
        self.mode = mode
        self.rho = 0.90
        self.STFTalpha = 1/winSize
        self.sigmaS = (winSize-1)/4
        self.sigmaA = 8/(winSize-1)
        self.convmode = 'valid'

    def compute_STFT_filters(self, x):
        if self.freqestim == 1:
            # Basic STFT filters
            w0 = np.ones_like(x)
            w1 = np.exp(-2*np.pi*x*self.STFTalpha*1j)
            w2 = np.conj(w1)
        return w0, w1, w2

    def compute_LPQdesc(self, img):
        img = np.float64(img)
        r = (self.winSize-1)/2
        x = np.arange(-r,r+1)[np.newaxis]
        w0, w1, w2 = self.compute_STFT_filters(x)

        # Run filters to compute the frequency response in the four points. Store real and imaginary parts separately
        # Run first filter
        filterResp1 = convolve2d(convolve2d(img,w0.T,self.convmode),w1,self.convmode)
        filterResp2 = convolve2d(convolve2d(img,w1.T,self.convmode),w0,self.convmode)
        filterResp3 = convolve2d(convolve2d(img,w1.T,self.convmode),w1,self.convmode)
        filterResp4 = convolve2d(convolve2d(img,w1.T,self.convmode),w2,self.convmode)

        # Initialize frequency domain matrix for four frequency coordinates (real and imaginary parts for each frequency).
        freqResp = np.dstack([filterResp1.real, filterResp1.imag,
                              filterResp2.real, filterResp2.imag,
                              filterResp3.real, filterResp3.imag,
                              filterResp4.real, filterResp4.imag])

        # Perform quantization and compute LPQ codewords
        inds = np.arange(freqResp.shape[2])[np.newaxis,np.newaxis,:]
        LPQdesc = ((freqResp>0)*(2**inds)).sum(2)

        # Switch format to uint8 if LPQ code image is required as output
        if self.mode == 'im':
            LPQdesc = np.uint8(LPQdesc)

        # Histogram if needed
        if self.mode == 'nh' or self.mode == 'h':
            LPQdesc = np.histogram(LPQdesc.flatten(),range(256))[0]

        # Normalize histogram if needed
        if self.mode == 'nh':
            LPQdesc = LPQdesc/LPQdesc.sum()

        return LPQdesc
    
    def __str__(self):
        return 'lpq'
    
if __name__ == "__main__":

    # Load image and convert to grayscale
    image = cv2.imread('/Users/melikapooyan/Downloads/BreaKHis_v1/breast/benign/SOB/adenosis/SOB_B_A_14-22549AB/40X/SOB_B_A-14-22549AB-40-003.png')
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Compute LPQ descriptor and histogram
    lpq = LPQ()
    lpq_desc = lpq.compute_LPQdesc(gray)
    hist = plt.hist(lpq_desc, bins=256)
    print (lpq_desc)
    print(hist[0])



