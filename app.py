from flask import Flask, request, render_template
import torch
import timm
from torchvision import transforms
from PIL import Image
import os

app = Flask(__name__)

# =========================================================
# 🔹 Device
# =========================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================================================
# 🔹 Model & Load Weights
# =========================================================
num_classes = 15  # Change if different
model = timm.create_model("resnet50", pretrained=True, num_classes=num_classes)
model.load_state_dict(torch.load("model.pth", map_location=device))
model.to(device)
model.eval()
print("Model loaded successfully")

# =========================================================
# 🔹 Transform for Input Images
# =========================================================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225]),
])

# =========================================================
# 🔹 Class Names
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
# 🔹 Flask Routes
# =========================================================
@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'file' not in request.files:
            return render_template('index.html', prediction_text="No file uploaded!")

        file = request.files['file']

        # Read image and preprocess
        img = Image.open(file.stream).convert('RGB')
        img = transform(img).unsqueeze(0).to(device)  # Add batch dimension

        # Predict
        with torch.no_grad():
            outputs = model(img)
            probs = torch.softmax(outputs, dim=1)
            confidence, pred_idx = torch.max(probs, dim=1)
            predicted_class = class_names[pred_idx.item()]

        return render_template(
            'index.html',
            prediction_text=f"Predicted: {predicted_class} (Confidence: {confidence.item()*100:.2f}%)"
        )

    except Exception as e:
        return render_template('index.html', prediction_text=f"Error: {e}")


# =========================================================
# 🔹 Run App
# =========================================================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)