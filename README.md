# Face Verification using Siamese Network with Triplet Loss (TensorFlow)

## 📌 Project Overview

This project implements a **Siamese neural network** for **face verification** — determining whether two face images belong to the same person.
The model is trained using **Triplet Loss**, which forces the network to learn a feature space where:

* embeddings of **the same person** are close together
* embeddings of **different people** are far apart

This type of system is commonly used in:

* biometric authentication
* identity verification
* security and access control
* face matching/search systems

All training, preprocessing, and evaluation are implemented inside a **single Jupyter notebook** for clarity and simplicity.

---

## 📂 Repository Structure

```
face-verification-siamese/
│
├── Reaidy.io ML Assignment.ipynb     # Main notebook
│
├── dataset/                          # Face dataset (e.g., LFW)
│   └── person_name/
│       └── image files...
│
└── model/
    └── face_siamese.h5               # Saved trained model
```

### Folder Details

#### 🧪 `Reaidy.io ML Assignment.ipynb`

This notebook contains everything:

* dataset loading
* face preprocessing
* model architecture
* triplet mining
* training loop
* evaluation (ROC curve & AUC)
* saving final model

So the project is easy to run and reproduce.

#### 🖼 `dataset/`

This folder contains the face images used for training and testing.
Each **sub-folder represents one person**, for example:

```
dataset/
 ├── Adam_Scott/
 ├── Kate_Winslet/
 ├── Elon_Musk/
```

This structure allows sampling **positive pairs (same person)** and **negative pairs (different people)**.

#### 🤖 `model/`

Contains the **trained Siamese embedding model**:

```
model/face_siamese.h5
```

You can reuse this model for inference later.

---

## 🧠 Approach & Method

### 1️⃣ Embedding Learning

Instead of directly predicting “same or different”, the network learns a **128-dimensional embedding vector** for each face.

### 2️⃣ Triplet Loss

Training uses **anchor, positive, negative** images:

* Anchor = reference image
* Positive = same person
* Negative = different person

The loss encourages:

```
distance(anchor, positive)   --> small
distance(anchor, negative)   --> large
```

Margin = 0.2

### 3️⃣ Backbone Network

The model uses **MobileNetV2** as a feature extractor:

* lightweight
* fast
* good accuracy

The final embedding is **L2-normalized**.

---

## 🛠 Technologies Used

* Python 3
* TensorFlow / Keras
* OpenCV
* NumPy
* scikit-learn
* Matplotlib

---

## 🚀 How to Run the Project

### Step 1 — Install Dependencies

Run:

```
pip install tensorflow opencv-python scikit-learn matplotlib numpy
```

### Step 2 — Place Dataset

Ensure your dataset is inside:

```
dataset/
```

with one folder per person.

### Step 3 — Open Notebook

Run:

```
Reaidy.io ML Assignment.ipynb
```

and execute cells in order.

---

## 📈 Model Evaluation

The notebook evaluates the model using:

### ✔ ROC Curve

Plots the trade-off between:

* True Positive Rate
* False Positive Rate

### ✔ AUC Score

Measures verification performance
(closer to **1.0 = better**)

Distance metric used:

```
Euclidean distance between embeddings
```

---

## 💾 Output Files

### 🧠 Trained Model

Saved to:

```
model/face_siamese.h5
```

This model converts face images → embeddings.

You can later:

* compare embeddings
* verify identity
* cluster people

---

## 🎯 Applications

* Face authentication systems
* Attendance tracking
* Identity verification
* Duplicate account detection
* Security systems

---

## 📌 Key Learning Outcomes

This project demonstrates:

✔ Deep metric learning
✔ Siamese architecture
✔ Triplet loss optimization
✔ Dataset preprocessing
✔ ROC-based evaluation
✔ Model export & reuse

All in a simple, reproducible setup.

---

## 👤 Author

Amruthaluri Gavin

---

## 📝 Notes

This project is for **educational and research purposes only**, not production biometric deployment.

---


