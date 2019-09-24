from preprocessing import *
from perform_ocr import *

def pan_parser():
	img = cv2.imread("/Users/aditya/klpl/TessaractTrainigModule/pan_new/pan45.png")
	print(determine_image_intensity(img))
	brighten_img = increase_brightness(img, 30)
	contrast_img = change_contrast(brighten_img)
	output = performOCR(brighten_img)
	print(output)
	cv2.imshow('bi', img)
	cv2.waitKey(0)
	
if __name__ == '__main__':
	pan_parser()
