# American-Sign-Language-
This project focuses on building a Convolutional Neural Network (CNN) model to recognize American Sign Language (ASL) hand signs. The model is trained using the Sign Language MNIST dataset and is deployed for real-time sign language prediction using a webcam feed.

Here's a description you can use for your GitHub repository:

---

# Sign Language Recognition with CNN and Real-Time Prediction

This project focuses on building a Convolutional Neural Network (CNN) model to recognize American Sign Language (ASL) hand signs. The model is trained using the Sign Language MNIST dataset and is deployed for real-time sign language prediction using a webcam feed.

## Key Features:

- **CNN Model:** A deep learning model built using Keras and TensorFlow, consisting of convolutional, max-pooling, batch normalization, dropout layers, and fully connected layers.
  
- **Dataset:** The model is trained on the Sign Language MNIST dataset, which contains 28x28 grayscale images of hand signs representing the letters A-Y (excluding J).
  
- **Model Training:** The model is trained with data augmentation techniques to enhance robustness. The final accuracy score is printed, and a confusion matrix is visualized for detailed performance analysis.
  
- **Real-Time Prediction:** The project uses OpenCV and MediaPipe for real-time hand detection and tracking. The trained CNN model predicts the sign language gesture in real time, which is displayed on the video feed.
  
- **Sign Language Prediction:** The project includes a Python script that captures live video from a webcam, processes the hand gestures, and predicts the corresponding ASL sign with the highest confidence level. The top three predictions are displayed with their confidence scores.

## Usage:

- **Model Training:** The `sign_mnist_train.csv` and `sign_mnist_test.csv` datasets are used to train the model. The trained model is saved as `smnist.h5`.

- **Real-Time Prediction:** The script captures real-time video using OpenCV, detects hand landmarks using MediaPipe, and predicts the ASL gesture using the trained CNN model.

## Dependencies:

- Python
- TensorFlow/Keras
- OpenCV
- MediaPipe
- Pandas
- Seaborn
- Matplotlib
- NumPy

## How to Run:

1. Clone the repository.
2. Ensure all dependencies are installed using `pip install -r requirements.txt`.
3. Train the model using the provided dataset or load the pre-trained model.
4. Run the real-time prediction script to start detecting and predicting ASL signs.

## License:

This project is licensed under the MIT License.
