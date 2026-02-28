import gradio as gr
import torch
import timm
from torchvision import transforms
from PIL import Image

# =========================================================
# Device
# =========================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================================================
# Model & Load Weights
# =========================================================
num_classes = 15
model = timm.create_model("resnet50", pretrained=True, num_classes=num_classes)
model.load_state_dict(torch.load("model.pth", map_location=device))
model.to(device)
model.eval()
print("Model loaded successfully")

# =========================================================
# Transform for Input Images
# =========================================================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

# =========================================================
# Class Names
# =========================================================
class_names = [
    "Pepper__bell___Bacterial_spot",
    "Pepper__bell___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Tomato_Bacterial_spot",
    "Tomato_Early_blight",
    "Tomato_Late_blight",
    "Tomato_Leaf_Mold",
    "Tomato_Septoria_leaf_spot",
    "Tomato_Spider_mites_Two_spotted_spider_mite",
    "Tomato__Target_Spot",
    "Tomato__Tomato_YellowLeaf__Curl_Virus",
    "Tomato__Tomato_mosaic_virus",
    "Tomato_healthy"
]

# =========================================================
# Prediction Function
# =========================================================
def predict(image: Image.Image):
    img = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(img)
        probs = torch.softmax(outputs, dim=1)
        confidence, pred_idx = torch.max(probs, dim=1)
        predicted_class = class_names[pred_idx.item()]
    return f"Predicted: {predicted_class} (Confidence: {confidence.item()*100:.2f}%)"

# =========================================================
# Gradio Interface for v4+
# =========================================================
with gr.Blocks() as demo:
    gr.Markdown("# 🌿 Plant Disease Detection")
    gr.Markdown("Upload an image of your plant leaf and the model will predict its disease.")
    
    with gr.Row():
        img_input = gr.Image(type="pil", label="Upload Plant Image")
        output = gr.Textbox(label="Prediction")
    
    predict_btn = gr.Button("Predict")
    predict_btn.click(fn=predict, inputs=img_input, outputs=output)

# =========================================================
# Launch App
# =========================================================
if __name__ == "__main__":
    demo.launch()