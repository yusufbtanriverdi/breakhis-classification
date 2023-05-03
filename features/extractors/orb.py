import cv2
import numpy as np

class ORB:
    def __init__(self, num_keypoints=500):
        self.num_keypoints = num_keypoints
        self.orb = cv2.ORB_create(nfeatures=num_keypoints)
    
    def describe(self, image):
        # Detect keypoints and compute their descriptors
        keypoints, descriptors = self.orb.detectAndCompute(image, None)
        
        # Compute the average descriptor if there are keypoints
        if len(keypoints) > 0:
            avg_descriptor = descriptors.mean(axis=0)
        else:
            avg_descriptor = np.zeros(32)
        
        return avg_descriptor
    
    def get_feature(self, image):
        return np.array(self.describe(image), dtype=np.float64)

if __name__ == "__main__":
    # Load an example image
    image = cv2.imread("/Users/melikapooyan/Downloads/BreaKHis_v1/breast/benign/SOB/adenosis/SOB_B_A_14-22549AB/40X/SOB_B_A-14-22549AB-40-007.png")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Initialize the ORB feature extractor
    orb_extractor = ORB(num_keypoints=500)
    
    # Extract features from the image
    features = orb_extractor.get_feature(gray)
    
    # Print the feature vector
    print(features)
