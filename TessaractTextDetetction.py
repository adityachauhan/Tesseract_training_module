from imutils.object_detection import non_max_suppression
import numpy as np
import argparse
import time
import cv2
import os
import pytesseract
import pandas as pd





ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", type = str, help = "path to input image")
ap.add_argument("-east", "--east", type = str, help = "path to input EAST text detetctor")
ap.add_argument("-c", "--min-confidence", type=float, default = 0.7, help="minimum probability required to inspect a region")
ap.add_argument("-w", "--width", type=int, default=320, help="resized image width (should be multiple of 32)")
ap.add_argument("-e", "--height", type=int, default=320, help="resized image height (should be multiple of 32)")
ap.add_argument("-p", "--padding", type=float, default=0.0, help="amount of padding to add to each border of ROI")
args = vars(ap.parse_args())

image = cv2.imread(args["image"])
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
# text = pytesseract.image_to_data(image, lang='plangclass+plangname+plangnum+plangpanum+plangsmall', output_type = 'data.frame')
# text = text[text.conf >= 70]
# # print(text)
# # print("***************************************************************************************************************")
# # print("***************************************************************************************************************")
# orig_top =[]
# orig_text = []

# for index, row in text.iterrows():
#     if row['text'].strip():
#         orig_top.append(row['top'])
#         orig_text.append(row['text'])
# # print(orig_top)
# # print(orig_text)

# diff = [y - x for x, y in zip(*[iter(orig_top)] * 2)]
# #print(diff)
# avg = sum(diff)/len(diff)
# m = [[orig_top[0]]]
# for x in orig_top[1:]:
#     if x - m[-1][0] < avg:
#         m[-1].append(x)
#     else:
#         m.append([x])


# #print(m)

# m_val = [[]]
# index_start = 0
# for i in range(0, len(m)):
#     index_end = len(m[i])
#     temp = []
#     for j in range(index_start, index_start + index_end):
#         temp.append(orig_text[j])
#     index_start = index_start + index_end 
#     m_val.append(temp)


# for i in range(0, len(m_val)):
#     print(m_val[i])

text = pytesseract.image_to_data(gray, lang='eng', output_type = 'data.frame')
print(text)
text = text[text.conf >= 40]
print(text)
print("***************************************************************************************************************")
print("***************************************************************************************************************")

orig_top =[]
orig_text = []

for index, row in text.iterrows():
    if row['text'].strip():
        orig_top.append(row['top'])
        orig_text.append(row['text'])
# print(orig_top)
# print(orig_text)

diff = [y - x for x, y in zip(*[iter(orig_top)] * 2)]
#print(diff)
avg = sum(diff)/len(diff)
m = [[orig_top[0]]]
for x in orig_top[1:]:
    if x - m[-1][0] < avg:
        m[-1].append(x)
    else:
        m.append([x])


#print(m)

m_val = [[]]
index_start = 0
for i in range(0, len(m)):
    index_end = len(m[i])
    temp = []
    for j in range(index_start, index_start + index_end):
        temp.append(orig_text[j])
    index_start = index_start + index_end 
    m_val.append(temp)


for i in range(0, len(m_val)):
    print(m_val[i])

# text = pytesseract.image_to_osd(image, lang='plangclass+plangname+plangnum+plangpanum+plangsmall')
# print(text)
# print("***************************************************************************************************************")
# print("***************************************************************************************************************")
# text = pytesseract.image_to_osd(gray, lang='plangclass+plangname+plangnum+plangpanum+plangsmall')
# print(text)
# print("***************************************************************************************************************")
# print("***************************************************************************************************************")
# #image = gray
# #cv2.imshow("Binary Image", gray)

# orig = image.copy()
# (origH, origW) = image.shape[:2]
# (H, W) = image.shape[:2]
# count = 0

# (newW, newH) = (args["width"], args["height"])
# rW = W / float(newW)
# rH = H / float(newH)
# image = cv2.resize(image, (newW, newH), interpolation =cv2.INTER_AREA)
# (H, W) = image.shape[:2]

# layerNames = ["feature_fusion/Conv_7/Sigmoid", "feature_fusion/concat_3"]
# # print("[INFO] loading EAST text detector...")
# net = cv2.dnn.readNet(args["east"])

# blob = cv2.dnn.blobFromImage(image, 1.0, (W, H), (123.68, 116.78, 103.94), swapRB=True, crop=False)

# start = time.time()
# net.setInput(blob)
# (scores, geometry) = net.forward(layerNames)
# end = time.time()


# # print("[INFO] text detection took {:.6f} seconds".format(end - start))
# (numRows, numCols) = scores.shape[2:4]
# rects = []
# confidences = []

# for y in range(0, numRows):
#     scoresData = scores[0, 0, y]
#     xData0 = geometry[0, 0, y]
#     xData1 = geometry[0, 1, y]
#     xData2 = geometry[0, 2, y]
#     xData3 = geometry[0, 3, y]
#     anglesData = geometry[0, 4, y]


#     for x in range(0, numCols):
#         if scoresData[x] < args["min_confidence"]:
#             continue
#         (offsetX, offsetY) = (x * 4.0, y * 4.0)
#         angle = anglesData[x]
#         cos = np.cos(angle)
#         sin = np.sin(angle)
#         h = xData0[x] + xData2[x]
#         w = xData1[x] + xData3[x]
#         endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
#         endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
#         startX = int(endX - w)
#         startY = int(endY - h)
#         rects.append((startX, startY, endX, endY))
#         confidences.append(scoresData[x])
# output = "output_new"
# boxes = non_max_suppression(np.array(rects), probs=confidences)
# config = ("-l eng --oem 1 --psm 7")
# results = []
# for (startX, startY, endX, endY) in boxes:
#     count = count + 1
#     startX = int(startX * rW)
#     startY = int(startY * rH)
#     endX = int(endX * rW)
#     endY = int(endY * rH)
#     dX = int((endX - startX) * args["padding"])
#     dY = int((endY - startY) * args["padding"])
#     startX = max(0, startX - dX)
#     startY = max(0, startY - dY)
#     endX = min(origW, endX + (dX))
#     endY = min(origH, endY + (dY*2))
#     #cv2.rectangle(orig, (startX, startY), (endX, endY), (255, 255, 255), 2)
#     img = orig[startY: endY, startX:endX]
#     # img = cv2.resize(img, (300, 300), interpolation =cv2.INTER_AREA)
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
#     text = pytesseract.image_to_data(gray, config=config, output_type = 'data.frame')
#     print(text)
#     text = text[text.conf >= 70]
#     #print(text)
#     # print("***************************************************************************************************************")
#     # print("***************************************************************************************************************")

#     # text = pytesseract.image_to_data(gray, config=config, output_type = 'data.frame')
#     # text = text[text.conf >= 70]
#     # print(text)
#     lines = text.groupby('block_num')['text'].apply(list)
#     # print("***************************************************************************************************************")
#     # print("***************************************************************************************************************")
#     #lines2 = text.groupby('block_num')['top'].apply(list)
#     #print(lines)
#     # print(lines2)
#     #print("===========================")
#     if(len(lines) != 0):
#         results.append(((startX, startY, endX, endY), lines))
#     #results.append(text)
#     fullpath = os.path.join(output, 'eng.arialfont.exp' + str(count) + '.png')
#     cv2.imwrite(fullpath, gray)

# results = sorted(results, key=lambda r:r[0][1])
# arr = []
# val = []
# #print(results[0][1])
# for i in range(0, len(results)):
#     s = str(results[i][1])
# #print(s)
#     ar = s.split()

#     for j in range(2, len(ar)-4):
#         arr.append(results[i][0][1])
#         # ar[j].replace('[', '').replace(']','').replace('.','').replace(',','')
#         val.append(ar[j])
#         #print(results[i][0][1], ar[j])

# #print(arr, val)
# # print(arr)
# # print("======================")
# # print(val)


# diff = [y - x for x, y in zip(*[iter(arr)] * 2)]
# avg = sum(diff)/len(diff)
# m = [[arr[0]]]
# for x in arr[1:]:
#     if x - m[-1][0] < avg:
#         m[-1].append(x)
#     else:
#         m.append([x])


# #print(m)

# m_val = [[]]
# index_start = 0
# for i in range(0, len(m)):
#     index_end = len(m[i])
#     temp = []
#     for j in range(index_start, index_start + index_end):
#         temp.append(val[j])
#     index_start = index_start + index_end 
#     m_val.append(temp)


# for i in range(0, len(m_val)):
#     print(m_val[i])



# #print(m_val)


# #
# # arr = []
# # i = 0
# # k = 0
# # for i in range(k, len(results)):
# #    val = results[i][0][1]
# #    print(val)
# #    temp= []
# #    temp.append(val)
# #    index = 0
# #    for j in range(k+1, len(results)):
# #        if results[j][0][1] <= val + 5 and results[j][0][1] >= val + 5:
# #            temp.append(results[j][1])
# #            index = j
# #    arr.append(temp)
# #    #print(k)
# #    k = index
# #    #print(k)
# # print(arr)

# # for ((startX, startY, endX, endY), text) in results:
# #     # display the text OCR'd by Tesseract
# #     print("OCR TEXT")
# #     print("========")
# #     print("{}\n".format(text))
# # #    print(startX)
# # #    print(startY)

# #     # strip out non-ASCII text so we can draw the text on the image
# #     # using OpenCV, then draw the text and a bounding box surrounding
# #     # the text region of the input image
# #     text = "".join([c if ord(c) < 128 else "" for c in text]).strip()
# #     output = orig.copy()
# #     cv2.rectangle(output, (startX, startY), (endX, endY),(0, 0, 255), 2)
# #     cv2.putText(output, text, (startX, startY - 20),cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
# #     # cv2.imshow("Text Detection", output)
# #     # cv2.waitKey(0)
# # #cv2.imshow("Text Detection", orig)
# # #cv2.waitKey(0)
