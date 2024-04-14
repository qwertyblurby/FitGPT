import torch
import torchvision.transforms as transforms
import torchvision.models as models
import numpy as np
import cv2
from PIL import Image

# Load the pre-trained Faster R-CNN model
model = models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
model.eval()

# Load the image
image = cv2.imread("image.jpg")

# Convert the image to RGB (PyTorch expects RGB format)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Convert image to tensor and normalize
transform = transforms.Compose([
    transforms.ToTensor(),
])
input_tensor = transform(image)
input_tensor = input_tensor.unsqueeze(0)  # Add batch dimension



# Perform inference
with torch.no_grad():
    prediction = model(input_tensor)

# Extract bounding box coordinates for the detected person
boxes = prediction[0]['boxes']  # Assuming only one prediction
scores = prediction[0]['scores']  # Confidence scores for each prediction
labels = prediction[0]['labels']  # Class labels for each prediction

# Filter out predictions for 'person' class (label 1)
person_boxes = boxes[labels == 1]

# Select the box with the highest confidence score
if len(person_boxes) > 0:
    max_score_index = torch.argmax(scores[labels == 1])
    person_box = person_boxes[max_score_index].detach().numpy().astype(int)
    xmin, ymin, xmax, ymax = person_box

    # Draw bounding box on the image
    # cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
    
    
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # adjust aspect ratio

    width = xmax - xmin
    height = ymax - ymin
    desired_ratio = 1 / 3
    aspect_ratio = width/height
    if aspect_ratio > desired_ratio: # more height
        extra_height = int(width / desired_ratio - height)
        ymin -= extra_height // 2
        ymax += (extra_height+1) // 2
    else: # more width
        extra_width = int(height * desired_ratio - width)
        xmin -= extra_width // 2
        xmax += (extra_width+1) // 2
    
    # Crop the image using the adjusted bounding box coordinates
    cropped_image = image[ymin:ymax, xmin:xmax]
    
    # Resize the cropped image to 200x600 pixels
    resized_image = cv2.resize(cropped_image, (200, 600))

    if input("Show image? ") == "yes":
        # Display the image
        cv2.namedWindow("Detected Person", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Detected Person", 400, 1200),
        cv2.imshow("Detected Person", resized_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        cv2.imwrite("output_image.png", resized_image)

else:
    print("No person detected in the image.")










'''




# Define the transformation
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1)  # Convert to grayscale
])

# Apply the transformation
grayscale_image = transform(image)

# Display or save the grayscale image
#grayscale_image.show()
grayscale_image.save("grayscale_image.png")

width, height = grayscale_image.size
 
# Setting the points for cropped image
left = 900
top = height / 20
right = 2000
bottom = 10 * height / 10
 
# Cropped image of above dimension
# (It will not change original image)
im1 = grayscale_image.crop((left, top, right, bottom))
 
# Shows the image in image viewer
im1.show()


'''







