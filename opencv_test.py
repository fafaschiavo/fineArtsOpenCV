# import sys
# sys.path.append('/usr/local/lib/python2.7/site-packages')

import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image

# reference - Vincent van Gogh (1853-1890) - Arbres dans le jardin de l'asile
# Best threshold - 30 ~ 40

# ======================================================
# Parameters here
matches_to_use = 20
# ======================================================

def get_compare_results(img, reference, matches_to_use = 20, minimum_reference_dimension = 800, display_results = False):
	reference_size = reference.shape
	new_size = reference.shape
	if reference_size[0] > reference_size[1]:
		min_size = reference_size[1]
	else:
		min_size = reference_size[0]

	resize_ratio = float(minimum_reference_dimension)/float(min_size)
	new_size = (int(reference.shape[1]*resize_ratio), int(reference.shape[0]*resize_ratio))

	# reference = cv2.cvtColor(reference,cv2.COLOR_BGR2RGB)
	# reference = Image.fromarray(reference)
	# reference.thumbnail(new_size, Image.ANTIALIAS).convert('RGB')
	# reference = reference[:, :, ::-1].copy() 

	reference = cv2.resize(reference, new_size, interpolation = cv2.INTER_AREA)

	orb = cv2.ORB_create()
	kp_reference, des_reference = orb.detectAndCompute(reference,None)
	kp, des = orb.detectAndCompute(img,None)

	bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

	matches = bf.match(des_reference, des)
	matches = sorted(matches, key = lambda x:x.distance)
	matches = matches[:matches_to_use]

	total_distance = 0
	for match in matches:
		total_distance = total_distance + match.distance

	print(total_distance/len(matches))

	if display_results:
		result = cv2.drawMatches(reference, kp_reference, img, kp, matches, None, flags=2)
		plt.imshow(result)
		plt.show()




reference = cv2.imread('reference.jpg',0)

img1 = cv2.imread('true_positive.jpg',0)
img2 = cv2.imread('true_positive_2.jpg',0)
img3 = cv2.imread('true_positive_3.jpg',0)

img4 = cv2.imread('false_positive.jpg',0)
img5 = cv2.imread('false_positive_2.jpg',0)
img6 = cv2.imread('false_positive_3.jpg',0)
img7 = cv2.imread('false_positive_4.jpg',0)

get_compare_results(img1, reference)
get_compare_results(img2, reference)
get_compare_results(img3, reference, display_results = False)
print('================')
get_compare_results(img4, reference)
get_compare_results(img5, reference)
get_compare_results(img6, reference)
get_compare_results(img7, reference)













# reference = cv2.imread('reference.jpg',0)

# img1 = cv2.imread('true_positive.jpg',0)
# img2 = cv2.imread('true_positive_2.jpg',0)
# img3 = cv2.imread('true_positive_3.jpg',0)

# img4 = cv2.imread('false_positive.jpg',0)
# img5 = cv2.imread('false_positive_2.jpg',0)
# img6 = cv2.imread('false_positive_3.jpg',0)
# img7 = cv2.imread('false_positive_4.jpg',0)

# kp_reference, des_reference = orb.detectAndCompute(reference,None)
# kp1, des1 = orb.detectAndCompute(img1,None)
# kp2, des2 = orb.detectAndCompute(img2,None)
# kp3, des3 = orb.detectAndCompute(img3,None)
# kp4, des4 = orb.detectAndCompute(img4,None)
# kp5, des5 = orb.detectAndCompute(img5,None)
# kp6, des6 = orb.detectAndCompute(img6,None)
# kp7, des7 = orb.detectAndCompute(img7,None)



# reference = cv2.resize(reference, (100, 100), interpolation = cv2.INTER_AREA)
# img1 = cv2.resize(img1, (100, 100), interpolation = cv2.INTER_AREA)



# img1_laplacian = cv2.Laplacian(img1,cv2.CV_8U)
# cv2.imwrite('laplacian.jpeg', img1_laplacian)

















# reference = cv2.calcHist([reference],[0],None,[256],[0,256])
# hist1 = cv2.calcHist([img1],[0],None,[256],[0,256])
# hist2 = cv2.calcHist([img2],[0],None,[256],[0,256])
# hist3 = cv2.calcHist([img3],[0],None,[256],[0,256])
# hist4 = cv2.calcHist([img4],[0],None,[256],[0,256])
# hist5 = cv2.calcHist([img5],[0],None,[256],[0,256])
# hist6 = cv2.calcHist([img6],[0],None,[256],[0,256])
# hist7 = cv2.calcHist([img7],[0],None,[256],[0,256])

# print('----------')

# result = cv2.compareHist(hist1,reference,cv2.HISTCMP_BHATTACHARYYA)
# print(result)

# result = cv2.compareHist(hist2,reference,cv2.HISTCMP_BHATTACHARYYA)
# print(result)

# result = cv2.compareHist(hist3,reference,cv2.HISTCMP_BHATTACHARYYA)
# print(result)

# result = cv2.compareHist(hist4,reference,cv2.HISTCMP_BHATTACHARYYA)
# print(result)

# result = cv2.compareHist(hist5,reference,cv2.HISTCMP_BHATTACHARYYA)
# print(result)

# result = cv2.compareHist(hist6,reference,cv2.HISTCMP_BHATTACHARYYA)
# print(result)

# result = cv2.compareHist(hist7,reference,cv2.HISTCMP_BHATTACHARYYA)
# print(result)

# print('----------')