import preprocessor
from preprocessor import cv2

FILECOUNT = 87
for i in range(11, FILECOUNT+1):
    image = cv2.imread("training_images/%s.jpg"%(i))
    try:
        preprocessor.preprocess(image, "training_set/%s.png"%(i))
        print("Processed image %s"%(i))
    except Exception as e:
        print("Error with image %s"%(i))
