import gradio as gr
from ultralytics import YOLO
import cv2
import numpy as np
import os

# Ensure we are in the correct directory (AquaEdge root)
model_path = "runs/detect/runs/aquaedge_newdata/weights/best.pt"

# Load the best trained model
try:
    print(f"Loading model from: {model_path}")
    model = YOLO(model_path)
except Exception as e:
    print(f"Error loading model: {e}")
    print("Make sure you run this script from the AquaEdge folder!")
    exit(1)

from collections import Counter

def predict_image(img, conf_threshold):
    if img is None:
        return None, ""
    
    # BUG FIX: Gradio sends images in RGB format, but YOLO expects BGR format (like OpenCV).
    # If we don't convert it first, the AI sees the colors inverted (blue water looks orange), 
    # and it won't detect anything!
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    
    # Run YOLO prediction with the correct BGR image
    results = model(img_bgr, conf=conf_threshold)
    
    # Get the image with bounding boxes drawn (YOLO returns BGR)
    res_img = results[0].plot()
    
    # Convert BGR back to RGB so Gradio displays the colors correctly
    res_img = cv2.cvtColor(res_img, cv2.COLOR_BGR2RGB)
    
    # Calculate the counts of each detected class
    detected_classes = results[0].boxes.cls.cpu().numpy()
    class_names = results[0].names
    
    if len(detected_classes) == 0:
        count_text = "No microorganisms detected."
    else:
        # Count occurrences of each class index
        counts = Counter(detected_classes)
        # Format the text output
        count_text = "Microorganism Counts:\n"
        for cls_id, count in counts.items():
            name = class_names[int(cls_id)]
            count_text += f"• {name}: {count}\n"
    
    return res_img, count_text

# Create a clean, simple web interface
iface = gr.Interface(
    fn=predict_image,
    inputs=[
        gr.Image(type="numpy", label="Upload Water Sample Image"),
        gr.Slider(minimum=0.01, maximum=1.0, value=0.51, step=0.01, label="Confidence Threshold (>50% Enforced)")
    ],
    outputs=[
        gr.Image(type="numpy", label="Detection Results"),
        gr.Textbox(label="Detected Counts")
    ],
    title="🔬 AquaEdge Model Testing Dashboard",
    description="Upload a sample microorganism image. If the AI is missing organisms, try lowering the Confidence Threshold slider!"
)

if __name__ == "__main__":
    print("Starting the temporary dashboard...")
    print("You will see a link like http://127.0.0.1:7860 below. Click it to open the dashboard!")
    iface.launch(share=False)
