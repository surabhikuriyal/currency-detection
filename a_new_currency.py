from utils import *
from matplotlib import pyplot as plt
import cv2


max_val = 8
max_pt = -1
max_kp = 0

sift = cv2.xfeatures2d.SIFT_create()
test_img = cv2.imread('test image path')

original = cv2.resize(test_img,None, fx=0.4, fy=0.4, interpolation=cv2.INTER_AREA)
display('original', original)

(kp1, des1) = sift.detectAndCompute(test_img,None)

training_set = ['path for all training images']

for i in range(0, len(training_set)):
	train_img = cv2.imread(training_set[i])

	(kp2, des2) = sift.detectAndCompute(train_img, None)

	bf = cv2.BFMatcher()
	all_matches = bf.knnMatch(des1, des2, k=2)

	good = []
	
	for (m, n) in all_matches:
		if m.distance < 0.789 * n.distance:
			good.append([m])

	if len(good) > max_val:
		max_val = len(good)
		max_pt = i
		max_kp = kp2

	print(i, ' ', training_set[i], ' ', len(good))

if max_val != 8:
	print(training_set[max_pt])
	print('good matches ', max_val)

	train_img = cv2.imread(training_set[max_pt])
	img3 = cv2.drawMatchesKnn(test_img, kp1, train_img, max_kp, good, 4)
	
	note = str(training_set[max_pt])[6:-4]
	print('\nDetected denomination: Rs. ', note)

	(plt.imshow(img3), plt.show())
else:
	print('No Matches')
