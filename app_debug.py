from flask import Flask, render_template, request, jsonify
import os
from model_old import color_order

app = Flask(__name__)

# Define the directory to store uploaded images
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
	os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

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
		file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
		probs_dict = {article: dict(zip(color_order, [0]*len(color_order))) for article in ("shirt", "outerwear", "pants", "shoes")}
		return jsonify({'message': 'File processed successfully', 'output': probs_dict}), 200

if __name__ == '__main__':
	app.run(debug=True)