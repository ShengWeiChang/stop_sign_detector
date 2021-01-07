'''
ECE276A WI20 HW1
Stop Sign Detector
'''

import numpy as np
import os, cv2
from skimage.measure import label, regionprops

color = 'RGB'

def load_image(imgBGR, color):
	# 1. Convert input image from BGR to RGB or YCrCb color space.
	# 2. Reshape input image into 3 by (m*n).

	if color == 'RGB':
		img = cv2.cvtColor(imgBGR, cv2.COLOR_BGR2RGB)
	elif color == 'YCbCr':
		img = cv2.cvtColor(imgBGR, cv2.COLOR_BGR2YCR_CB)

	pixels_array = np.reshape(img, (-1,3))
	return pixels_array

def gaussian(x, mu, sig):
	# The multivariate Gaussian distribution
	L = np.linalg.cholesky(np.linalg.inv(sig))
	expTerm = np.exp(-np.sum(np.square(np.dot(x-mu, L)), axis = 1) / 2)
	denominator = np.sqrt((2*np.pi)**3 * np.linalg.det(sig))
	return expTerm / denominator

class StopSignDetector():
	def __init__(self):
		'''
			Initilize your stop sign detector with the attributes you need,
			e.g., parameters of your classifier
		'''

		# All colors, all data
		if color == 'RGB':
			'''
			self.mu_blue = np.array([ 106.32696045284762 , 148.32640286906891 , 201.0777186347703 ])
			self.mu_brwn = np.array([ 139.20916654606165 , 103.78299936072575 , 69.64969718192108 ])
			self.mu_orng = np.array([ 169.14539134204188 , 116.81573704348986 , 80.89911553113981 ])
			self.mu_othr = np.array([ 107.50309236401463 , 109.62905178793655 , 102.10582157175617 ])
			self.mu_ored = np.array([ 115.36591420932713 , 57.02063770101015 , 61.38112578856563 ])
			self.mu_sred = np.array([ 161.3237125352304 , 15.32027143124588 , 26.35305860237528 ])
		
			self.cov_blue = np.array([ [2388.69912323, 1913.3432948 , 1026.14991707] , [1913.3432948,  1824.97852504, 1123.98347644] , [1026.14991707, 1123.98347644,  952.17631369] ])
			self.cov_brwn = np.array([ [4239.25522742, 3138.90543003, 1877.20749456] , [3138.90543003, 2731.64375814, 2053.14389894] , [1877.20749456, 2053.14389894, 2026.4131502 ] ])
			self.cov_orng = np.array([ [2623.23266988, 2572.15074572, 1295.30603522] , [2572.15074572, 3102.9567313,  1738.07035391] , [1295.30603522, 1738.07035391, 1546.98883444] ])
			self.cov_othr = np.array([ [3313.42772962, 3099.95140829, 3047.00924281] , [3099.95140829, 3098.95577588, 3065.13104889] , [3047.00924281, 3065.13104889, 3485.05804823] ])
			self.cov_ored = np.array([ [2833.5241197 ,  997.57205282,  944.21663791] , [ 997.57205282, 1432.21680477, 1408.44684507] , [ 944.21663791, 1408.44684507, 1492.63410039] ])
			self.cov_sred = np.array([ [2898.92219889,  320.23672272,  492.01529842] , [320.23672272, 562.88347196, 519.56228942] , [492.01529842, 519.56228942, 635.32527426] ])
		
			self.prior_blue = 0.2568322410412888
			self.prior_brwn = 0.06575565674834602
			self.prior_orng = 0.0018280167650162158
			self.prior_othr = 0.6595718676206456
			self.prior_ored = 0.0011570577026393705
			self.prior_sred = 0.014855160122063957
			'''

			self.mu_blue = np.array([ 112.26013365296106 , 154.38224470490147 , 205.68737086505868 ])
			self.mu_brwn = np.array([ 132.62528405240352 , 99.77635395882915 , 71.35151265725759 ])
			self.mu_orng = np.array([ 169.07883735117878 , 114.05221140566577 , 77.73615836336893 ])
			self.mu_othr = np.array([ 110.60507493417744 , 112.27153361620427 , 106.39803824909035 ])
			self.mu_ored = np.array([ 88.84016420100492 , 55.11897978619418 , 55.96442072313218 ])
			self.mu_sred = np.array([ 161.32589750953923 , 15.328712612316908 , 26.36139272965987 ])
		
			self.cov_blue = np.array([ [2228.39363126, 1799.02126704, 1014.33187336] , [1799.02126704, 1716.93380557, 1109.20069224] , [1014.33187336, 1109.20069224,  968.20919344] ])
			self.cov_brwn = np.array([ [3872.13082968, 2852.14123252, 1727.88710486] , [2852.14123252, 2456.05263493, 1841.01284083] , [1727.88710486, 1841.01284083, 1864.29640919] ])
			self.cov_orng = np.array([ [2705.02254407, 2464.44221998, 1207.74499615] , [2464.44221998, 2982.29144797, 1708.32326126] , [1207.74499615, 1708.32326126, 1612.63278495] ])
			self.cov_othr = np.array([ [3550.39855876, 3356.87805056, 3314.64838217] , [3356.87805056, 3324.42226794, 3300.74491792] , [3314.64838217, 3300.74491792, 3602.61753733] ])
			self.cov_ored = np.array([ [1476.8278114 ,  689.21192928,  677.72520982] , [689.21192928, 836.11859708, 825.27729758] , [677.72520982, 825.27729758, 888.78119595] ])
			self.cov_sred = np.array([ [2898.7883877 ,  320.49533896,  492.25909714] , [320.49533896, 563.77530602, 520.44028303] , [492.25909714, 520.44028303, 636.18183231] ])
		
			self.prior_blue = 0.18777379592808147
			self.prior_brwn = 0.044917514741610945
			self.prior_orng = 0.001076530585644033
			self.prior_othr = 0.7493939436160402
			self.prior_ored = 0.008981942846658679
			self.prior_sred = 0.007856272281964652

		elif color == 'YCbCr':
			self.mu_blue = np.array([ 141.29746152679343 , 102.23218174211125 , 160.9739510327226 ])
			self.mu_brwn = np.array([ 109.96982873545251 , 147.9708871242381 , 104.44122854210595 ])
			self.mu_orng = np.array([ 127.88658625030219 , 156.553680840954 , 100.76030143632407 ])
			self.mu_othr = np.array([ 107.64361964897604 , 127.06249073533785 , 124.15068045362374 ])
			self.mu_ored = np.array([ 74.4458178034743 , 156.34314293616404 , 119.82499440266565 ])
			self.mu_sred = np.array([ 59.73449336168008 , 199.66688995288933 , 108.39387176771395 ])
		
			self.cov_blue = np.array([ [1746.51501119,  148.52229545, -378.82252643] , [ 148.52229545,  115.58194649, -103.19296924] , [-378.82252643, -103.19296924,  174.76893919] ])
			self.cov_brwn = np.array([ [2851.1101882 ,  337.97005579, -481.24018783] , [ 337.97005579,  225.02422689, -237.81218506] , [-481.24018783, -237.81218506,  279.45137351] ])
			self.cov_orng = np.array([ [2546.83638096,  -74.42065941, -544.91180366] , [-74.42065941, 146.09302444, -73.49158218] , [-544.91180366,  -73.49158218,  296.21980691] ])
			self.cov_othr = np.array([ [3116.50246958,   30.06821496,   -5.06050625] , [ 30.06821496,  57.64391872, -41.55857218] , [ -5.06050625, -41.55857218, 123.69791227] ])
			self.cov_ored = np.array([ [1369.70479259,  122.04874244,  -50.63190993] , [ 122.04874244,  570.45062239, -203.16294694] , [ -50.63190993, -203.16294694,   96.55779484] ])
			self.cov_sred = np.array([ [676.87491473, 309.84853386, -85.78432267] , [ 309.84853386,  690.80032953, -188.12755726] , [ -85.78432267, -188.12755726,   84.2052173 ] ])
			
			self.prior_blue = 0.2568322410412888
			self.prior_brwn = 0.06575565674834602
			self.prior_orng = 0.0018280167650162158
			self.prior_othr = 0.6595718676206456
			self.prior_ored = 0.0011570577026393705
			self.prior_sred = 0.014855160122063957

	def segment_image(self, img):
		'''
			Obtain a segmented image using a color classifier,
			e.g., Logistic Regression, Single Gaussian Generative Model, Gaussian Mixture, 
			call other functions in this class if needed
			
			Inputs:
				img - original image
			Outputs:
				mask_img - a binary image with 1 if the pixel in the original image is red and 0 otherwise
		'''
		
		image_size = np.shape(img)
		test = load_image(img, color)
		
		# Claculate the Gaussian probability
		p_blue = gaussian(test, self.mu_blue, self.cov_blue) * self.prior_blue
		p_brwn = gaussian(test, self.mu_brwn, self.cov_brwn) * self.prior_brwn
		p_orng = gaussian(test, self.mu_orng, self.cov_orng) * self.prior_orng
		p_othr = gaussian(test, self.mu_othr, self.cov_othr) * self.prior_othr
		p_ored = gaussian(test, self.mu_ored, self.cov_ored) * self.prior_ored
		p_sred = gaussian(test, self.mu_sred, self.cov_sred) * self.prior_sred
		
		# Choose the max probability and output a binary mask
		max_p = list(map(max, zip(p_blue, p_brwn, p_orng, p_othr, p_ored, p_sred)))
		mask_img = 255*np.array(max_p == p_sred).astype('uint8')
		
		# Closing (morphology), erode first and dilate
		kernel = np.ones((10,10), np.uint8)
		mask_img = cv2.morphologyEx(mask_img, cv2.MORPH_CLOSE, kernel)
		
		# Opening (morphology), erode first and dilate
		kernel = np.ones((4,4), np.uint8)
		mask_img = cv2.morphologyEx(mask_img, cv2.MORPH_OPEN, kernel)
		
		mask_img = mask_img.reshape(image_size[:2])
		return mask_img

	def get_bounding_box(self, img):
		'''
			Find the bounding box of the stop sign
			call other functions in this class if needed
			
			Inputs:
				img - original image
			Outputs:
				boxes - a list of lists of bounding boxes. Each nested list is a bounding box in the form of [x1, y1, x2, y2] 
				where (x1, y1) and (x2, y2) are the top left and bottom right coordinate respectively. The order of bounding boxes in the list
				is from left to right in the image.
				
			Our solution uses xy-coordinate instead of rc-coordinate. More information: http://scikit-image.org/docs/dev/user_guide/numpy_images.html#coordinate-conventions
		'''
		
		image_size = np.shape(img)
		mask_img = self.segment_image(img)
		
		boxes = []
		labels = label(mask_img)
		regions = regionprops(labels)
		for r in regions:
			score = 0
			minr, minc, maxr, maxc = r.bbox
			h = maxr - minr
			w = maxc - minc
			a = r.area
			e = r.extent
			er = r.euler_number
			
			# Aspect ratio
			if h/w > 0.5 and h/w < 2:
				score = score + 2
			if h/w > 0.9 and h/w < 1.2:
				score = score + 3
			
			# Area proportion
			if e > 0.45 and e < 0.85:
				score = score + 3
			if a > 9000 and e > 0.55 and e < 0.75:
				score = score + 3   
			if a > 500 and a < 9000 and e > 0.75 and e < 0.85:
				score = score + 3
			
			# Euler Number
			if er < 0:
				score = score + 5;
			if score > 10:
				boxes.append([minc, image_size[0]-maxr, maxc, image_size[0]-minr])
		
		# Sort from left to right
		sorted(boxes, key = lambda x:(x[0]))
		
		return boxes
	
if __name__ == '__main__':

	folder = "trainset"
	my_detector = StopSignDetector()
	for filename in os.listdir(folder):
		# read one test image
		img = cv2.imread(os.path.join(folder,filename))
		# cv2.imshow('image', img)
		# cv2.waitKey(1)
		# cv2.destroyAllWindows()

		mask_img = my_detector.segment_image(img)
		boxes = my_detector.get_bounding_box(img)
	
	'''	
		print(boxes)
		print(filename)
		print("-----------------")

		image_size = np.shape(img)
		box_img = []
		for i in range(len(boxes)):
			box_img = cv2.rectangle(img, (boxes[i][0],image_size[0]-boxes[i][1]), (boxes[i][2],image_size[0]-boxes[i][3]), (0, 255, 0) , 2)

		cv2.imshow('mask', mask_img)
		cv2.waitKey(3)
		cv2.imwrite(os.path.join("result_mask", filename), mask_img)

		if len(box_img) > 0:
			cv2.imshow('box', box_img)
			cv2.waitKey(1)
			cv2.imwrite(os.path.join("result_box", filename), box_img)
		else:
			cv2.imwrite(os.path.join("result_box", filename), image)
		
	cv2.waitKey(1)
	cv2.destroyAllWindows()
	'''

		#Display results:
		#(1) Segmented images
		#	 mask_img = my_detector.segment_image(img)
		#(2) Stop sign bounding box
		#	boxes = my_detector.get_bounding_box(img)
		#The autograder checks your answers to the functions segment_image() and get_bounding_box()
		#Make sure your code runs as expected on the testset before submitting to Gradescope

