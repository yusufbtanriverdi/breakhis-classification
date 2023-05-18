# from __future__ import division

# import numpy as np
# from scipy.signal import convolve2d
# import cv2
# import matplotlib.pyplot as plt

# class LPQ:
#     def __init__(self, winSize=3, freqestim=1, mode='nh'):
#         self.winSize = winSize
#         self.freqestim = freqestim
#         self.mode = mode
#         self.rho = 0.90
#         self.STFTalpha = 1/winSize
#         self.sigmaS = (winSize-1)/4
#         self.sigmaA = 8/(winSize-1)
#         self.convmode = 'valid'

#     def compute_STFT_filters(self, x):
#         if self.freqestim == 1:
#             # Basic STFT filters
#             w0 = np.ones_like(x)
#             w1 = np.exp(-2*np.pi*x*self.STFTalpha*1j)
#             w2 = np.conj(w1)
#         return w0, w1, w2



#     def describe(self, img):
#         img = np.float64(img)
#         r = (self.winSize-1)/2
#         x = np.arange(-r,r+1)[np.newaxis]
#         w0, w1, w2 = self.compute_STFT_filters(x)
#         # print(w0.shape)
#         # print(w1.shape)
#         # print(w2.shape)
#         # print(img.shape)
        

#         # Run filters to compute the frequency response in the four points. Store real and imaginary parts separately
#         # Run first filter
#         filterResp1 = convolve2d(convolve2d(img,w0.T,self.convmode),w1,self.convmode)
#         filterResp2 = convolve2d(convolve2d(img,w1.T,self.convmode),w0,self.convmode)
#         filterResp3 = convolve2d(convolve2d(img,w1.T,self.convmode),w1,self.convmode)
#         filterResp4 = convolve2d(convolve2d(img,w1.T,self.convmode),w2,self.convmode)

#         # Initialize frequency domain matrix for four frequency coordinates (real and imaginary parts for each frequency).
#         freqResp = np.dstack([filterResp1.real, filterResp1.imag,
#                               filterResp2.real, filterResp2.imag,
#                               filterResp3.real, filterResp3.imag,
#                               filterResp4.real, filterResp4.imag])

#         # Perform quantization and compute LPQ codewords
#         inds = np.arange(freqResp.shape[2])[np.newaxis,np.newaxis,:]
#         LPQdesc = ((freqResp>0)*(2**inds)).sum(2)
#         # print(filterResp1.shape)
#         # print(filterResp2.shape)
#         # print(filterResp3.shape)
#         # # print(filterResp4.shape)
#         # print(img.shape)
#         # print(w0.T.shape)
#         # filterResp1 = convolve2d(convolve2d(img,w0.T,self.convmode),w1,self.convmode)
        
        


#         # Switch format to uint8 if LPQ code image is required as output
#         if self.mode == 'im':
#             LPQdesc = np.uint8(LPQdesc)

#         # Histogram if needed
#         if self.mode == 'nh' or self.mode == 'h':
#             LPQdesc = np.histogram(LPQdesc.flatten(),range(256))[0]

#         # Normalize histogram if needed
#         if self.mode == 'nh':
#             LPQdesc = LPQdesc/LPQdesc.sum()

#         return LPQdesc
    
#     def __str__(self):
#         return 'lpq'
# import numpy as np
# from scipy.signal import convolve2d

# class LPQ:
#     def __init__(self, winSize=3, mode='nh'):
#         self.winSize = winSize
#         self.mode = mode

#     def _get_filters(self):
#         r = (self.winSize - 1) // 2
#         x = np.arange(-r, r+1)
#         u = np.arange(1, r+1)

#         w0 = np.ones_like(x)
#         w1 = np.exp(-2j * np.pi * x / self.winSize)
#         w2 = np.conj(w1)

#         return w0, w1, w2

#     def _compute_response(self, img):
#         w0, w1, w2 = self._get_filters()

#         filterResp = convolve2d(convolve2d(img, w0[:, None], mode='valid'), w1, mode='valid')
#         freqResp = np.zeros((filterResp.shape[0], filterResp.shape[1], 8))
#         freqResp[..., 0] = np.real(filterResp)
#         freqResp[..., 1] = np.imag(filterResp)
#         filterResp = convolve2d(convolve2d(img, w1[:, None], mode='valid'), w0, mode='valid')
#         freqResp[..., 2] = np.real(filterResp)
#         freqResp[..., 3] = np.imag(filterResp)
#         filterResp = convolve2d(convolve2d(img, w1[:, None], mode='valid'), w1, mode='valid')
#         freqResp[..., 4] = np.real(filterResp)
#         freqResp[..., 5] = np.imag(filterResp)
#         filterResp = convolve2d(convolve2d(img, w1[:, None], mode='valid'), w2, mode='valid')
#         freqResp[..., 6] = np.real(filterResp)
#         freqResp[..., 7] = np.imag(filterResp)

#         return freqResp

#     def _quantize(self, freqResp):
#         freqRow, freqCol, freqNum = freqResp.shape

#         LPQdesc = np.zeros((freqRow, freqCol))
#         for i in range(freqNum):
#             LPQdesc += (freqResp[..., i] > 0) * (2 ** i)

#         return LPQdesc

#     def _normalize_histogram(self, LPQdesc):
#         hist, _ = np.histogram(LPQdesc.ravel(), bins=np.arange(256))
#         LPQdesc = hist / np.sum(hist)

#         return LPQdesc

#     def compute(self, img):
#         if len(img.shape) != 2:
#             raise ValueError('Only gray scale image can be used as input')
#         if self.winSize < 3 or self.winSize % 2 != 1:
#             raise ValueError('Window size winSize must be odd number and greater than equal to 3')
#         if self.mode not in ['nh', 'h', 'im']:
#             raise ValueError('mode must be nh, h, or im. See help for details.')

#         img = img.astype(float)

#         freqResp = self._compute_response(img)
#         LPQdesc = self._quantize(freqResp)

#         if self.mode == 'im':
#             LPQdesc = LPQdesc.astype(np.uint8)

#         if self.mode == 'nh' or self.mode == 'h':
#             LPQdesc = self._normalize_histogram(LPQdesc)

#         return LPQdesc

    
# if __name__ == "__main__":

#     # Load image and convert to grayscale
#     image = cv2.imread('/Users/melikapooyan/Documents/BreaKHis_v1/breast/benign/SOB/adenosis/SOB_B_A_14-22549AB/40X/SOB_B_A-14-22549AB-40-003.png')
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#     # Compute LPQ descriptor and histogram
#     lpq = LPQ()
#     lpq_desc = lpq.describe(gray)
#     hist = plt.hist(lpq_desc, bins=256)
#     print (lpq_desc)
#     print(hist[0])

import cv2
import numpy as np

class LPQ:
    def __init__(self, radius=3, neighbors=8):
        self.radius = radius
        self.neighbors = neighbors

    def describe(self, image):
        eps = 1e-7
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        lbp = np.zeros(gray.shape, dtype=np.uint8)

        for i in range(self.radius, gray.shape[0] - self.radius):
            for j in range(self.radius, gray.shape[1] - self.radius):
                center = gray[i, j]
                code = 0

                for k in range(self.neighbors):
                    angle = 2 * np.pi * k / self.neighbors
                    x = i + int(round(self.radius * np.cos(angle)))
                    y = j - int(round(self.radius * np.sin(angle)))
                    val = gray[x, y]

                    if val >= center:
                        code |= 1 << k

                lbp[i, j] = code

        hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, 257), range=(0, 256))
        hist = hist.astype("float")
        hist /= (hist.sum() + eps)
        return hist


if __name__ == "__main__":
    # Load image
    image = cv2.imread('/Users/melikapooyan/Documents/BreaKHis_v1/breast/benign/SOB/adenosis/SOB_B_A_14-22549AB/40X/SOB_B_A-14-22549AB-40-003.png')

    # Compute LPQ descriptor
    lpq = LPQ()
    lpq_desc = lpq.describe(image)

    print(lpq_desc)







