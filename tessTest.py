from PIL import Image
import pytesseract
import argparse
import cv2
import os
import numpy as np
import re


ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="path to input image to be OCR'd")
args = vars(ap.parse_args())
image = cv2.imread(args["image"])
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
text = pytesseract.image_to_data(gray, lang='plangclass+plangname+plangnum+plangpanum+plangsmall')
print(text)
#newStr = " ".join(text.split())#text.replace(" ", "")
#val = newStr.split()
#print(val)
#index = 0
#for i in range(len(val)):
#    if val[i].find('/') != -1:
#        index = i
#print(index)
#dob = ""
#if re.search(r'\d', val[index + 1]):
#    dob = val[index] + val[index+1]
#print(dob)


#def hasNumbers(inputString):
   #return bool(re.search(r'\d', inputString))
cv2.imshow("Output", gray)
cv2.waitKey(0)
