from PIL import Image
import cv2
import numpy as np
import time



def binarize_image(img):
	image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
	return image

def inverse_image(img):
	image = cv2.bitwise_not(img)
	return image

def increase_brightness(img, value):
	hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
	h,s,v = cv2.split(hsv)
	lim = 255 - value
	v[v > lim] = 255
	v[v <= lim] += value

	final_hsv = cv2.merge((h,s,v))
	img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
	return img

def change_contrast(img):
	lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
	l,a,b = cv2.split(lab)
	clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
	cl = clahe.apply(l)
	limg = cv2.merge((cl,a,b))
	image = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
	return image


def determine_image_intensity(img):
	image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	blur = cv2.blur(image, (5, 5))
	val = cv2.mean(blur)
	if val > 127:
		return 'light'
	else:
		return 'dark'



