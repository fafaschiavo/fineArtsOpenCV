# import sys
# sys.path.append('/usr/local/lib/python2.7/site-packages')

import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import requests
from io import BytesIO

def get_images_feature_matching_score(image_url, reference_url, matches_to_use = 20, minimum_reference_dimension = 800, display_results = False):
	response = requests.get(reference_url)
	img_array = np.array(bytearray(response.content), dtype=np.uint8)
	reference = cv2.imdecode(img_array, -1)
	# reference = cv2.imread(BytesIO(response.content),0)

	response = requests.get(image_url)
	img_array = np.array(bytearray(response.content), dtype=np.uint8)
	img = cv2.imdecode(img_array, -1)
	# img = cv2.imread(BytesIO(response.content),0)

	reference_size = reference.shape
	new_size = reference.shape
	if reference_size[0] > reference_size[1]:
		min_size = reference_size[1]
	else:
		min_size = reference_size[0]


	resize_ratio = float(minimum_reference_dimension)/float(min_size)
	new_size = (int(reference.shape[1]*resize_ratio), int(reference.shape[0]*resize_ratio))

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

	return total_distance/len(matches)

image_url = 'https://www.christies.com/img/LotImages/2019/CKS/2019_CKS_18168_0011_000(howard_hodgkin_as_time_goes_by).jpg'
reference_url = 'https://s3.eu-west-3.amazonaws.com/fine-arts-static/product-images/ip2pusuiarwuk83tumu7msd6f2racyw7cb07iwuf3aucqj7nsty7ajs6a7m01y90/aben3tpviu74y1xtb7lg2s6yf723zx0h1yedyonjtiudjiz82k8cup9853nt5vre.jpeg'

print get_images_feature_matching_score(image_url, reference_url)