import cv2
from PIL import Image
from preprocessing import *
import pytesseract
import pandas as pd



def performOCR(img):
	ocr_text = pytesseract.image_to_string(img, lang='eng')
	# text = pytesseract.image_to_data(img, lang='plangclass+plangname+plangnum+plangpanum+plangsmall', output_type = 'data.frame')
	# text = text[text.conf >= 40]
	# orig_top =[]
	# orig_text = []

	# for index, row in text.iterrows():
	# 	if row['text'].strip():
	# 		orig_top.append(row['top'])
	# 		orig_text.append(row['text'])

	# diff = [y - x for x, y in zip(*[iter(orig_top)] * 2)]
	
	# avg = sum(diff)/len(diff)
	# m = [[orig_top[0]]]
	# for x in orig_top[1:]:
	#     if x - m[-1][0] < avg:
	#         m[-1].append(x)
	#     else:
	#         m.append([x])


	

	# m_val = [[]]
	# index_start = 0
	# for i in range(0, len(m)):
	#     index_end = len(m[i])
	#     temp = []
	#     for j in range(index_start, index_start + index_end):
	#         temp.append(orig_text[j])
	#     index_start = index_start + index_end 
	#     m_val.append(temp)

	return ocr_text
