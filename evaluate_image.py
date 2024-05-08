import preprocessor
from inference_model import MyModel

def main():
	IMAGENAME = input("Path to image to be processed in uploads folder (ex. image.png): ")
	image = cv2.imread(f"uploads/{IMAGENAME}")
	preprocessed_path = f"uploads_processed/{IMAGENAME}"
	preprocessor.preprocess(image, preprocessed_path)
	print("Processed image!")
	
	model = MyModel()
	model.load_state_dict(torch.load("fitgpt_model.pt"))
	model.eval()
	
	transform = transforms.Compose([
		transforms.Resize((200, 600)),
		transforms.Grayscale(num_output_channels=1),
		transforms.ToTensor(),
	])
	
	image = Image.open(preprocessed_path)
	image = transform(image)
	with torch.no_grad():
		shirt_output, outerwear_output, pants_output, shoes_output = model(image)
	
	for article, article_output in (
		("Shirt", shirt_output),
		("Outerwear", outerwear_output),
		("Pants", pants_output),
		("Shoes", shoes_output)):
		print(f"{article} probabilities:")
		probs = list(zip(color_order, article_output[0].tolist()))
		probs.sort(key = lambda x: x[1])
		for color, prob in probs:
			print(f"{color}: {round(100*prob)}%")

if __name__ == "__main__":
	main()
