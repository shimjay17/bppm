import cv2
import numpy as np

class Image:
    def __init__(self, path: str):
        self.path = path
        self.image = cv2.imread(path)
       
        self.keypoints = None
        self.features = None
        self.H = np.eye(3)
        self.component_id: int = 0
        self.gain = np.ones(3, dtype=np.float32)

    def compute_features(self):
        #descriptor = cv2.SIFT_create() #SIFT use
        descriptor = cv2.ORB_create() #ORB use
        keypoints, features = descriptor.detectAndCompute(self.image, None)
        self.keypoints = keypoints
        self.features = features
