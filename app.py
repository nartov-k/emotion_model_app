from flask import Flask, render_template, request, jsonify
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import io
import base64
#import cv2
import numpy as np

app = Flask(__name__)

# Load your trained model
model = models.resnet50(pretrained=False)  # adjust this according to your model
model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False) #make sure we have 1-chane as input for conv layer
num_ftrs = model.fc.in_features
num_classes = 5
model.fc = torch.nn.Linear(num_ftrs, num_classes)  # adjust this

model_path = r"extra-version3_resnet50_fc_layer4_layer3_tuned_kritika_data_5classes_1channel-images.pth"
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
model.eval()
class_names = ['anger', 'fear', 'happy', 'neutral', 'sad']

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided.'}), 400

    file = request.files['file']
    img_bytes = file.read()
    img = Image.open(io.BytesIO(img_bytes))
    
    # Your image processing here
    transformation = transforms.Compose([
        transforms.Resize((270, 270)),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize([0.485], [0.229])
    ])
    img_tensor = transformation(img).unsqueeze(0)

    with torch.no_grad():
        outputs = model(img_tensor)
        _, predicted = torch.max(outputs.data, 1)
        predicted_class = class_names[predicted.item()]

    return jsonify({'emotion': predicted_class})

if __name__ == '__main__':
    app.run(debug=True)


# if 'file' not in request.files:
#     return 'No file part', 400
# file = request.files['file']
# if file.filename == '':
#     return 'No selected file', 400
