# Traffic-Sign-Classification-Using-Deep-Learning
# 🚦 Traffic Sign Classification (GTSRB)

This project implements a **Traffic Sign Recognition System** using the **German Traffic Sign Recognition Benchmark (GTSRB)** dataset.  
It classifies traffic signs into **43 categories** using a **Convolutional Neural Network (CNN)** built with TensorFlow/Keras.

---

## 📂 Dataset
- Dataset: [GTSRB - German Traffic Sign Recognition Benchmark](https://benchmark.ini.rub.de/gtsrb_dataset.html)  
- Training labels are provided in `Train.csv`.  
- Each class corresponds to a specific traffic sign (e.g., Stop, Yield, Speed limit signs).  

---

## ⚙️ Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/yourusername/traffic-sign-classification.git
cd traffic-sign-classification

# Recommended: create virtual environment
conda create -n traffic python=3.10
conda activate traffic

# Install dependencies
pip install -r requirements.txt
requirements.txt

makefile
Copy code
numpy==1.26.4
pandas
matplotlib
seaborn
opencv-python
tensorflow
scikit-learn
🧑‍💻 How to Run
Place dataset inside Downloads/task8 (or update path in script).

Run training script:

bash
Copy code
python traffic_sign_classifier.py
The model will:

Preprocess images (resize 32x32, normalize)

Train CNN with data augmentation

Evaluate on test set

Show accuracy, loss, and predictions

🏗️ Model Architecture
Conv2D(32) + MaxPooling2D

Conv2D(64) + MaxPooling2D

Conv2D(128) + MaxPooling2D

Flatten → Dense(256) → Dropout(0.5)

Dense(43, softmax)

📊 Results
Achieved ~95%+ accuracy on test data (depending on epochs & hyperparameters).

Training/Validation accuracy & loss plots are saved.

Example predictions:

Prediction	Ground Truth
Stop (14)	Stop (14)
Yield (13)	Yield (13)
Speed 60	Speed 60

🛑 Classes Overview
Some traffic signs included:

Speed limit (20–120 km/h)

Stop

Yield

No entry

Pedestrian crossing

Road work

Turn left/right ahead

Roundabout mandatory

(Full list of 43 classes is included in the code.)

📌 Future Improvements
Try Transfer Learning (e.g., MobileNet, ResNet).

Deploy as a Flask Web App or Streamlit App.

Export model to TensorFlow Lite for mobile/embedded systems.

