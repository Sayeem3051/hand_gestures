
# Hand Sign Digit Recognition (1–9) using OpenCV, MediaPipe & CNN

This project implements a deep learning-based system that recognizes hand sign digits (1 to 9) using computer vision techniques. It utilizes MediaPipe for hand landmark detection and a custom-trained 7-layer Convolutional Neural Network (CNN) for classification.

## 📂 Dataset

- The dataset is organized into `train`, `valid`, and `test` folders.
- Each folder contains subdirectories labeled from `0` to `9` (representing the digits).
- Input images are assumed to be of size `100x100` pixels, though only landmarks are used for training.

## 🧠 Technologies Used

- **OpenCV**: Image processing and webcam input.
- **MediaPipe**: Hand landmark detection (21 points per hand).
- **TensorFlow/Keras**: Model creation and training.
- **NumPy** and **sklearn**: Data handling and preprocessing.

## 📌 Project Structure

```
hand-sign-recognition/
├── train.py           # Training script
├── predict.py         # Real-time prediction using webcam
├── hand_sign_cnn_model.h5  # Saved trained model
├── dataset/
│   ├── train/
│   ├── valid/
│   └── test/
├── README.md
└── Hand_Sign_Project_Presentation.pptx
```

## 🏗️ Model Architecture (7-layer CNN)

- Input: Flattened hand landmark (21 points × 3 = 63)
- 3 × Conv1D + MaxPooling1D layers
- Dense + Dropout layers
- Output: 10-class softmax (digits 0–9)

## 🚀 How to Run

### 1. Install Dependencies

```bash
pip install opencv-python mediapipe tensorflow numpy scikit-learn
```

### 2. Train the Model

```bash
python train.py
```

### 3. Run Real-Time Prediction

```bash
python predict.py
```

## ✅ Results

- **Validation Accuracy:** 97%
- **Inference:** Real-time digit prediction using webcam input.
- **Robustness:** Accurate under standard lighting and clear backgrounds.

## 📈 Future Improvements

- Extend recognition to full ASL alphabets.
- Optimize with TensorFlow Lite for mobile apps.
- Add GUI for user interaction.
- Improve performance in varying lighting/backgrounds.

## 📚 References

- [MediaPipe](https://google.github.io/mediapipe/)
- [TensorFlow](https://www.tensorflow.org/)
- [OpenCV](https://docs.opencv.org/)
