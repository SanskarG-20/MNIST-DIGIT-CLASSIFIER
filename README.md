# MNIST Digit Classifier

**A fully interactive MNIST Digit Classifier** built with Streamlit. This app allows users to draw digits, upload images, or capture from a webcam for real-time handwritten digit recognition (0-9). Powered by a Convolutional Neural Network (CNN), it features animated confidence bars, dark mode support, and a modern, intuitive UI for a seamless experience.

## Live Demo

https://mnist-digit-classifier-69.streamlit.app

## Features

- **Drawing Canvas**: Draw digits freely with a mouse or touch input
- **Image Upload**: Upload PNG/JPG images of handwritten digits
- **Webcam Capture**: Use your camera to capture digits in real-time
- **Real-Time Predictions**: Instant classification with animated confidence bars for all 10 digits
- **Dark Mode Support**: Automatic theme switching for better usability
- **Modern UI**: Clean, responsive design with smooth animations and user feedback

## Screenshots

*(Recommended: Add actual screenshots of the app interface, drawing canvas, prediction results, etc.)*

## Tech Stack

- **Framework**: Streamlit (for the interactive web app)
- **Deep Learning**: Keras/TensorFlow (CNN model trained on MNIST dataset)
- **Model Format**: `.keras` (pre-trained model included)
- **Language**: Python

## Project Structure

```
MNIST-DIGIT-CLASSIFIER/
├── app.py                  # Main Streamlit application script
├── mnist_model.keras       # Pre-trained CNN model for digit classification
├── requirements.txt        # Python dependencies
├── README.md               # This documentation
```

## Prerequisites

- Python 3.8 or higher
- Git (for cloning)

## Installation and Setup

### 1. Clone the Repository

```bash
git clone https://github.com/SanskarG-20/MNIST-DIGIT-CLASSIFIER.git
cd MNIST-DIGIT-CLASSIFIER
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

Typical dependencies include:
- streamlit
- tensorflow (or keras)
- numpy
- pillow
- opencv-python (for webcam support)

### 3. Run the Application

```bash
streamlit run app.py
```

The app will launch in your default browser at `http://localhost:8501`.

## Usage

1. Open the app.
2. Choose an input method:
   - **Draw**: Use the canvas to draw a digit.
   - **Upload**: Select an image file.
   - **Webcam**: Allow camera access and capture a digit.
3. The model will instantly predict the digit with confidence scores displayed as animated bars.
4. Clear the canvas or recapture as needed.

## Model Details

- **Dataset**: Classic MNIST handwritten digits (60,000 training + 10,000 test images)
- **Architecture**: Convolutional Neural Network (CNN) with layers like Conv2D, MaxPooling, Dropout, and Dense
- **Accuracy**: Typically ~99% on the test set (standard for well-trained MNIST models)
- **Input**: 28x28 grayscale images (preprocessed automatically in the app)

The pre-trained model (`mnist_model.keras`) is included for immediate use—no training required!

## Deployment

Easily deploy to:
- **Streamlit Community Cloud**: Connect your GitHub repo and deploy for free.
- **Heroku / Render / Vercel**: With minor adaptations.

## Contributing

Contributions are welcome!  
- Fork the repo
- Create a feature branch
- Submit a pull request

Ideas for improvements:
- Add model training script/notebook
- Support for multi-digit recognition
- Enhanced preprocessing options

## Acknowledgments

- MNIST dataset by Yann LeCun et al.
- Built with Streamlit and TensorFlow communities
- Inspired by classic deep learning tutorials

Enjoy classifying digits! ✍️🔢
