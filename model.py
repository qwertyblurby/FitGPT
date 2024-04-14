import torchvision.transforms as transforms
from PIL import Image

# Load your image
image = Image.open("input_image.jpg")

# Define the transformation
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1)  # Convert to grayscale
])

# Apply the transformation
grayscale_image = transform(image)

# Display or save the grayscale image
grayscale_image.show()
# grayscale_image.save("grayscale_image.jpg")
