import torch
from inference_model import MyModel
from model_old import color_order

def main():
	model = MyModel(len(color_order))
	model.load_state_dict(torch.load("fitgpt_model.pt"))
	model.eval()
	dummy_input = torch.zeros(1, 200, 600)
	torch.onnx.export(model, dummy_input, "fitgpt_model_onnx.onnx", verbose=True)

if __name__ == "__main__":
	main()
