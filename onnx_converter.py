import torch
from inference_model import MyModel

def main():
	model = MyModel()
	model.load_state_dict(torch.load("fitgpt_model.pt"))
	model.eval()
	dummy_input = torch.zeros(200 * 600)
	torch.onnx.export(model, dummy_input, "fitgpt_model_onnx.onnx", verbose=True)

if __name__ == "__main__":
	main()
