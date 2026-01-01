# Bollywood Celebrity Predictor

This project is a deep learning-based application that predicts the name of a Bollywood celebrity from a given image. It leverages a powerful combination of a state-of-the-art face detection model and a pre-trained image classification model to deliver accurate predictions.

The application first detects and extracts faces from an input image using the **MTCNN** model and then uses the **VGG16** model to classify the cropped face against a dataset of Bollywood celebrities.

## Project Demo: https://drive.google.com/file/d/1U9DjfMkFmEWIaLUt83x7FL59lkT5jEwQ/view?usp=sharing

## ✨ Features

* **Face Detection:** Accurately detects and localizes faces in an image using the robust **MTCNN** (Multi-task Cascaded Convolutional Networks) model.
* **Face Alignment & Cropping:** Automatically crops the detected face region, ensuring that only the relevant features are fed into the classification model.
* **Celebrity Prediction:** Predicts the name of the celebrity from the cropped face using a fine-tuned **VGG16** model.
* **Scalable Architecture:** The modular workflow allows for easy swapping of models or expansion of the celebrity dataset.
* **Intuitive Workflow:** A clear, step-by-step process from an input image to a final prediction.

---

## 🛠️ Tech Stack

* **Python:** The core programming language for the entire project.
* **Deep Learning Framework:**
    * **TensorFlow / Keras:** For building, training, and running the deep learning models.
* **Face Detection:**
    * **MTCNN:** A multi-task cascaded convolutional network for accurate face and landmark detection. It's known for its high performance in real-world scenarios.
* **Image Classification:**
    * **VGG16:** A pre-trained Convolutional Neural Network (CNN) used as a feature extractor. The model, originally trained on the ImageNet dataset, is fine-tuned on a custom Bollywood celebrity dataset.
* **Data Handling & Image Processing:**
    * **OpenCV:** For image loading, manipulation, and visualization.
    * **NumPy:** For efficient numerical operations.
    * **Scikit-learn:** For data preprocessing and model evaluation metrics.

---

## 🔄 Workflow

The project follows a two-stage pipeline for celebrity prediction:

1.  **Face Detection and Preprocessing:**
    * An input image is provided to the system.
    * The **MTCNN** model scans the image to identify all faces.
    * For each detected face, a bounding box is generated.
    * The face region within the bounding box is cropped and resized to a fixed input size (e.g., 224x224 pixels) suitable for the VGG16 model.

2.  **Celebrity Prediction:**
    * The cropped face image is fed into the pre-trained **VGG16** model.
    * The VGG16 model, with its final layers replaced, extracts a high-dimensional feature vector (embedding) from the face image.
    * This feature vector is then passed to a custom classification head (e.g., a softmax layer) that has been trained on a dataset of Bollywood celebrity faces.
    * The model outputs a prediction, indicating the celebrity it most likely is, along with a confidence score.
