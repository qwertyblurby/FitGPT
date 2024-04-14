import preprocessor
from preprocessor import cv2
preprocessor.preprocess(cv2.imread("training_images/13.jpg"), "training_set/13.png")

'''
FILECOUNT = 87
for i in range(11, FILECOUNT+1):
    image = cv2.imread("training_images/%s.jpg"%(i))
    preprocessor.preprocess(image, "training_set/%s.png"%(i))
    print("Processed image %s"%(i))'''