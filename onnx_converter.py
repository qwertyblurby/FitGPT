import torch
from inference_model import MyModel
from model_old import color_order

def main():
	model = MyModel(len(color_order))
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	model.load_state_dict(torch.load("fitgpt_model.pt", map_location=device))
	model.eval()
	dummy_input = torch.zeros(1, 200, 600)
	torch.onnx.export(model, dummy_input, "fitgpt_model_onnx.onnx")

if __name__ == "__main__":
	main()
