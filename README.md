🌿 **Plant Disease Detection Using Deep Learning**

📌 **Project Overview**

Plant diseases significantly affect agricultural productivity and food security worldwide. Early and accurate identification of plant diseases helps farmers take timely preventive measures and reduce crop loss.

This project presents a **Deep Learning–based Plant Disease Detection System** that automatically classifies plant leaf images into healthy or diseased categories using computer vision techniques.

The system was deployed as an **interactive web application** using **Gradio** and hosted on **Hugging Face Spaces**, allowing real-time disease prediction from uploaded leaf images.

 🚀 **Live Demo**

🔗 Hugging Face Application:
https://huggingface.co/spaces/Ghazia0601/Plant_Disease_Detection

Users can upload a plant leaf image and instantly receive the predicted disease class.

🎯 **Objectives**

* Automate plant disease identification using Deep Learning
* Compare performance of modern CNN and Transformer architectures
* Build an end-to-end deployable AI application
* Provide real-time predictions through a web interface

🧠 **Models Implemented**

1️⃣ Vision Transformer (ViT)

Initially, a **Vision Transformer (ViT)** model was implemented to explore transformer-based image classification performance.
Although ViT demonstrated strong learning capability, it required higher computational resources and longer training time.

2️⃣ ResNet50 (Final Model )

The final model selected for deployment was **ResNet50**, a deep Convolutional Neural Network known for residual learning.
Reasons for choosing ResNet50:

* Faster convergence
* Better generalization
* Stable training performance
* Higher validation accuracy

✅ **Final Model Accuracy: ~99%**
🗂 Dataset

The model was trained on a labeled plant leaf dataset containing multiple plant disease categories including healthy and infected leaves.
Dataset preprocessing included:
* Image resizing
* Normalization
* Data augmentation
* Train–validation split

## ⚙️ Tech Stack

| Category         | Tools & Libraries            |
| ---------------- | ---------------------------- |
| Programming      | Python                       |
| Deep Learning    | PyTorch                      |
| Models           | ResNet50, Vision Transformer |
| Image Processing | Pillow, NumPy                |
| Deployment       | Gradio                       |
| Hosting          | Hugging Face Spaces          |
| Version Control  | Git & GitHub                 |


 🏗 **Project Workflow**

1. Data preprocessing and augmentation
2. Model training using Vision Transformer
3. Performance comparison with ResNet50
4. Model evaluation and accuracy analysis
5. Selection of best-performing model
6. Model serialization (`model.pth`)
7. Web app development using Gradio
8. Deployment on Hugging Face Spaces

## 📊 Model Performance

| Model                | Accuracy |
| -------------------- | -------- |
| Vision Transformer   | ~98%     |
| **ResNet50 (Final)** | **~99%** |

## 🖥 Application Features

* Upload plant leaf image
* Real-time disease prediction
* Simple and user-friendly interface
* Cloud-hosted AI application
* No installation required
  

## ▶️ Running Locally

## 🌍 Future Improvements
* Multi-crop disease detection
* Explainable AI visualization (Grad-CAM)
* Larger real-world agricultural datasets
* Edge-device deployment

## 👩‍💻 Author

**Ghazia**
Data Science Enthusiast | Machine Learning Developer

* GitHub: https://github.com/ghazia0601
* Hugging Face: https://huggingface.co/Ghazia0601

## ⭐ Acknowledgment

This project was developed as part of independent learning and practical exploration of Deep Learning and Computer Vision techniques for real-world agricultural applications.
