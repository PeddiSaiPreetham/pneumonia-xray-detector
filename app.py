# app.py
import torch
import numpy as np
import cv2
import gradio as gr
from model_def import get_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
model = get_model()
model.load_state_dict(torch.load("pneumonia_detector.pt", map_location=device))
model.to(device)
model.eval()

def predict(image):
    # Convert PIL image to OpenCV format
    img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    img = cv2.resize(img, (224, 224))
    img = img.astype(np.float32) / 255.0
    img = (img - 0.5) / 0.5
    img = np.transpose(img, (2, 0, 1))
    img_tensor = torch.tensor(img, dtype=torch.float32).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(img_tensor)
        prob = torch.softmax(output, dim=1)[0]
        class_idx = torch.argmax(prob).item()
        classes = ['Normal', 'Pneumonia']
        confidence = prob[class_idx].item()

    return {classes[0]: float(prob[0]), classes[1]: float(prob[1])}

# Gradio UI
interface = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs=gr.Label(num_top_classes=2),
    title="Chest X-ray Pneumonia Classifier",
    description="Upload a chest X-ray image to detect pneumonia using a ResNet18 model."
)

if __name__ == "__main__":
    interface.launch()
