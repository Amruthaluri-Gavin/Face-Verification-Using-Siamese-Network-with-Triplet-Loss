# Face Verification using Siamese Network with Triplet Loss (TensorFlow)

## 📌 Project Overview

This project implements a **Siamese neural network** for **face verification** — checking whether two face images belong to the same person.

The model is trained using **Triplet Loss**, which teaches the network to:

* pull **same-person images closer** in embedding space
* push **different-person images further apart**

This is the same idea used in biometric identity systems such as **FaceNet**.

👉 **Important Note**
The **dataset is NOT uploaded to this repository**.
Instead, the dataset is **downloaded automatically inside the notebook** (e.g., from Kaggle/LFW) when you run it.
This keeps the repo small and avoids dataset licensing issues.

---

## 📂 Repository Structure

```
face-verification-siamese/
│
├── Reaidy.io ML Assignment.ipynb     # Main notebook (training + evaluation)
│
└── Models/
    └── face_siamese.h5               # Saved trained model
```

That’s all you need in the repo.

The dataset will be downloaded at runtime to a local folder such as:

```
dataset/
```

but that folder is not committed to GitHub.

---

## 🧠 What This Project Does

The notebook performs the full workflow:

### 1️⃣ Download dataset (automatically)

* Downloads a public face dataset (e.g., LFW)
* Extracts images
* Organizes them by person

### 2️⃣ Preprocess images

* Resize
* Normalize
* Convert to RGB

### 3️⃣ Build Siamese embedding model

* Uses MobileNetV2 backbone
* Adds 128-D embedding layer
* Applies L2-normalization

### 4️⃣ Train using Triplet Loss

With triplets:

* Anchor (A)
* Positive (P)
* Negative (N)

Loss encourages:

```
distance(A,P) + margin < distance(A,N)
```

### 5️⃣ Evaluate performance

* Compute embeddings for face pairs
* Measure distances
* Plot ROC curve
* Compute AUC score

### 6️⃣ Save trained model

Exports model to:

```
model/face_siamese.h5
```

---

## 🛠 Tools & Libraries Used

* Python 3
* TensorFlow / Keras
* OpenCV
* NumPy
* scikit-learn
* Matplotlib

---

## 🚀 How To Run This Project

### ✔ Step 1 — Install dependencies

```
pip install tensorflow opencv-python scikit-learn matplotlib numpy kaggle
```

(if Kaggle is used)

### ✔ Step 2 — Open the notebook

```
Reaidy.io ML Assignment.ipynb
```

### ✔ Step 3 — Run all cells

The notebook will:

✅ download the dataset
✅ train the model
✅ evaluate it
✅ save model to `model/face_siamese.h5`

No manual dataset upload is needed 🎉

---

## 📈 Evaluation

The notebook reports:

### ROC Curve

Shows verification performance

### AUC Score

Measures accuracy
(higher = better)

Distance metric used:

```
Euclidean distance between embeddings
```

---

## 💾 Output

### Trained Embedding Model

Saved as:

```
model/face_siamese.h5
```

You can reuse it for:

* face authentication
* identity verification
* embedding visualization

---

## 🎯 Real-World Applications

* Login authentication
* Attendance systems
* Duplicate detection
* Person recognition
* Smart security

---

## 📚 Key Concepts Demonstrated

✔ Siamese neural networks
✔ Metric learning
✔ Triplet loss optimization
✔ Online triplet sampling
✔ ROC-AUC evaluation
✔ Exportable embedding models

---

## 👤 Author

Amruthaluri Gavin

---

## ⚠️ Disclaimer

This project is for **educational & research purposes only** —
not for deployment in real-world biometric security systems.

---

