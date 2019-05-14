import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image

# reference - Vincent van Gogh (1853-1890) - Arbres dans le jardin de l'asile

# reference = cv2.imread('control_reference.jpg',0)
# img1 = cv2.imread('control_true_positive.jpg',0)

reference = cv2.imread('reference.jpg',0)
img1 = cv2.imread('true_positive.jpg',0)

orb = cv2.ORB_create()

kp_reference, des_reference = orb.detectAndCompute(reference,None)
kp1, des1 = orb.detectAndCompute(img1,None)

bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

matches = bf.match(des_reference, des1)
matches = sorted(matches, key = lambda x:x.distance)



img3 = cv2.drawMatches(reference, kp_reference, img1, kp1, matches[:10], None, flags=2)
plt.imshow(img3)
plt.show()