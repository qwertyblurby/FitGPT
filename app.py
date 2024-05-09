from flask import Flask, render_template, request, jsonify
import os
import torchvision.transforms as transforms
from PIL import Image
import onnxruntime
from preprocessor import preprocess
from model_old import color_order

app = Flask(__name__)

# Define the directory to store uploaded images
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
PROCESSED_FOLDER = 'uploads_processed'
if not os.path.exists(PROCESSED_FOLDER):
    os.makedirs(PROCESSED_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['PROCESSED_FOLDER'] = PROCESSED_FOLDER

transform = transforms.Compose([
    transforms.Resize((200, 600)),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
])

# Load model
model = onnxruntime.InferenceSession("fitgpt_model_onnx.onnx")

# Define a route to render the HTML form
@app.route('/')
def index():
    return render_template('index.html')

# Define a route to handle the file upload
@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'message': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'message': 'No selected file'}), 400
    if file:
        filename = file.filename
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        # Preprocess image
        preprocessed_image_path = os.path.join(app.config['PROCESSED_FOLDER'], filename)
        preprocess(file_path, preprocessed_image_path)
        
        # Load preprocessed image
        image = Image.open(preprocessed_image_path)
        image = transform(image).numpy()
        
        # Evaluate with ONNX model
        outputs = model.run(None, {'onnx::Reshape_0': image})
        
        # Process outputs
        probs_dict = {article: dict(zip(color_order, map(float, article_output[0]))) for article, article_output in zip(("shirt", "outerwear", "pants", "shoes"), outputs)}
        return jsonify({'message': 'File processed successfully', 'output': probs_dict}), 200

if __name__ == '__main__':
    app.run(debug=True)
