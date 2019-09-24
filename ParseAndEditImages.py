#from PIL import Image
#import pytesseract
#import argparse
#import cv2
#import os
#import numpy as np
#
#def main():
#    outpath = "Passport_Back_binarized"
#    path = "Passport_Back"
#    count = 0
#    for image_path in os.listdir(path):
#        if not image_path.startswith('.'):
#            print(count)
#            input_path = os.path.join(path, image_path)
#
#            print(input_path)
#            image = cv2.imread(input_path)
#            print(image.size)
#            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#            gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
#
#            fullpath = os.path.join(outpath, 'eng.arialfont.exp'+str(count) + '.png')
#            cv2.imwrite(fullpath, gray)
#            count = count + 1
#
#if __name__ == '__main__':
#    main()


from PIL import Image
import pytesseract
import argparse
import cv2
import os
import numpy as np

def main():
    outpath = "Passport_Back_Trainig_300_res"
    path = "Passport_Back_Trainig"
    count = 0
    for image_path in os.listdir(path):
        if not image_path.startswith('.'):
            print(count)
            input_path = os.path.join(path, image_path)

            print(input_path)
            im = Image.open(input_path)
#            image = cv2.imread(input_path)
#            print(image.size)
#            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#            gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
#

            fullpath = os.path.join(outpath, image_path)
            im.save(fullpath, dpi=(300,300))
#            count = count + 1

if __name__ == '__main__':
    main()



#######################################################################################################################################
#                                                                                                                                     #
#   Terminal script to run for loop for makinf box files from png files for tesseract training                                        #
#                                                                                                                                     #
#   for f in /Users/aditya/klpl/TessaractTrainigModule/output_images/*; do tesseract $f ${f%.*} batch.nochop makebox; done            #
#                                                                                                                                     ###########################################
# ./textcleaner -g -e stretch -f 30 -o 10 -t 60 -u -s 1 -T -p 20 /Users/aditya/klpl/TessaractTrainigModule/pan83.jpeg /Users/aditya/klpl/TessaractTrainigModule/testing.png     #
#                                                                                                                                     ###########################################
#                                                                                                                                     #
#######################################################################################################################################
