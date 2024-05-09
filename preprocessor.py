import torch
import torchvision.transforms as transforms
import torchvision.models as models
import numpy as np
import cv2
from PIL import Image
from rembg import remove 

# Load the pre-trained Faster R-CNN model
model = models.detection.fasterrcnn_resnet50_fpn(weights="FasterRCNN_ResNet50_FPN_Weights.COCO_V1")
model.eval()

def preprocess(image_path, processed_path):
    image = cv2.imread(image_path)
    
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
        
        # Crop the image using the adjusted bounding box coordinates and resize to 200x600
        image = cv2.resize(image[ymin:ymax, xmin:xmax], (200, 600))

        # Remove background
        image = remove(Image.fromarray(resized_image))
        
        # Define the transformation
        transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1)  # Convert to grayscale
        ])

        # Apply the transformation
        image = transform(image)

        # Convert to CV2
        preprocessed_image = cv2.cvtColor(np.array(image), cv2.COLOR_GRAY2BGR)
        
        # Save image
        cv2.imwrite(processed_path, preprocessed_image)
    
    else:
        print("No person detected in image %s."%(processed_path))
