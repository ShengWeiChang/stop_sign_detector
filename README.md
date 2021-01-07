# Stop Sign Detector

UCSD ECE 276 Sensing & Estimation in Robotics
2020 Winter
Course Project 1: Stop Sign Detector

In this project, I design an algorithm to detect the stop sign.

I applied the Gaussian naive Bayes to segment unknown images, detect stop signs in an image, and draw bounding boxes. Firstly, we labeled the 200 training images and trained Gaussian models in RGB and YCbCr color spaces for image segmentation. Secondly, I processed segmented images by morphology operations (closing and opening) to remove noises. Lastly, I designed a method based on the Euler characteristic, the aspect ratio, and area proportion to score the similaritybetween regions in segmented images and a stop sign shape.