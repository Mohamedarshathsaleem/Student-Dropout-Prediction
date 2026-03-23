# 🎓 Student Dropout Prediction

A machine learning project that predicts student academic outcomes — **Dropout**, **Enrolled**, or **Graduate** — using a **Multi-Layer Perceptron (MLP)** neural network. Built as a Jupyter Notebook using Python and scikit-learn/TensorFlow, this project demonstrates how deep learning can be applied to educational data to identify at-risk students early.

---

## 📋 Table of Contents

- [About](#about)
- [Dataset](#dataset)
- [Problem Statement](#problem-statement)
- [Approach](#approach)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [Results](#results)
- [Key Features Used](#key-features-used)
- [References](#references)

---

## About

Student dropout is a significant challenge in higher education, carrying personal, institutional, and societal costs. This project applies a **Multi-Layer Perceptron (MLP)** — a type of feedforward neural network — to classify students into one of three outcome categories based on demographic, academic, and socioeconomic attributes collected at enrollment.

The goal is to enable educational institutions to identify students at risk of dropping out early enough to intervene with targeted support.

---

## 📊 Dataset

This project uses the **Predict Students' Dropout and Academic Success** dataset from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success).

| Property        | Details                                            |
|-----------------|----------------------------------------------------|
| **Source**      | UCI ML Repository (Dataset ID: 697)               |
| **Instances**   | ~4,424 students                                    |
| **Features**    | 36 input features                                  |
| **Target**      | 3 classes: `Dropout`, `Enrolled`, `Graduate`       |
| **Domain**      | Higher Education (undergraduate students)          |
| **License**     | CC BY 4.0                                          |

### Downloading the Dataset

You can load the dataset directly via the `ucimlrepo` Python package:

```python
from ucimlrepo import fetch_ucirepo

dataset = fetch_ucirepo(id=697)
X = dataset.data.features
y = dataset.data.targets
```

Or download it manually from the [UCI Repository page](https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success).

---

## 🎯 Problem Statement

Given a set of features known at the time of student enrollment, predict which of three outcomes a student will have:

- **Dropout** — student leaves the program before completion
- **Enrolled** — student is still ongoing (mid-course)
- **Graduate** — student successfully completes the program

This is a **multi-class classification** problem with class imbalance, as the Graduate class tends to dominate.

---

## 🧠 Approach

The notebook follows a standard machine learning pipeline:

1. **Data Loading** — Load and inspect the UCI dataset
2. **Exploratory Data Analysis (EDA)** — Understand distributions, correlations, and class balance
3. **Preprocessing**
   - Handle missing values
   - Encode categorical variables
   - Feature scaling (StandardScaler / MinMaxScaler)
   - Address class imbalance (e.g., oversampling / class weights)
4. **Model: Multi-Layer Perceptron (MLP)**
   - Fully connected neural network with multiple hidden layers
   - Activation functions: ReLU (hidden), Softmax (output)
   - Trained using backpropagation with gradient descent
   - Regularisation via dropout layers or L2 penalty
5. **Evaluation**
   - Accuracy, Precision, Recall, F1-Score
   - Confusion Matrix
   - ROC-AUC (per class)
6. **Insights** — Key features driving dropout predictions

---

## 🛠️ Tech Stack

| Tool / Library      | Purpose                          |
|---------------------|----------------------------------|
| Python 3.x          | Core language                    |
| Jupyter Notebook    | Development environment          |
| Pandas              | Data manipulation                |
| NumPy               | Numerical operations             |
| Matplotlib / Seaborn| Data visualisation               |
| Scikit-learn        | Preprocessing, metrics, MLP      |
| ucimlrepo           | Dataset loading                  |

---

## 📁 Project Structure

```
Student-Dropout-Prediction/
├── studentoutcomes_mlp.ipynb   # Main notebook: full ML pipeline
└── README.md
```

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/Mohamedarshathsaleem/Student-Dropout-Prediction.git
cd Student-Dropout-Prediction
```

### 2. Create a Virtual Environment (Recommended)

```bash
python -m venv venv
source venv/bin/activate        # macOS/Linux
venv\Scripts\activate           # Windows
```

### 3. Install Dependencies

```bash
pip install jupyter pandas numpy matplotlib seaborn scikit-learn ucimlrepo
```

Or if a `requirements.txt` is present:

```bash
pip install -r requirements.txt
```

### 4. Launch Jupyter Notebook

```bash
jupyter notebook
```

Then open `studentoutcomes_mlp.ipynb` in your browser and run all cells (`Kernel > Restart & Run All`).

---

## 📈 Results

The MLP model is evaluated on a held-out test set (typically an 80/20 train-test split). Key metrics reported:

| Metric      | Description                                   |
|-------------|-----------------------------------------------|
| Accuracy    | Overall proportion of correct predictions     |
| Precision   | Correctness of positive predictions per class |
| Recall      | Ability to find all instances of each class   |
| F1-Score    | Harmonic mean of precision and recall         |
| Confusion Matrix | Visual breakdown of predicted vs actual  |

> **Note:** Refer to the notebook for the exact model accuracy and per-class performance metrics on this dataset.

---

## 🔑 Key Features Used

The dataset includes 36 features across three categories:

**Demographic & Socioeconomic**
- Age at enrollment
- Gender
- Nationality
- Parental education level and occupation
- Scholarship holder status
- Debtor / tuition fees up to date

**Academic (Enrollment Time)**
- Course enrolled
- Previous qualification and grade
- Admission grade
- Displaced / international student status

**Academic Performance (Semesters 1 & 2)**
- Curricular units credited, enrolled, evaluated, approved
- Grade averages per semester
- Attendance patterns

**Macroeconomic Indicators**
- Unemployment rate
- Inflation rate
- GDP

---

## 📚 References

- Realinho, V., Vieira Martins, M., Machado, J., & Baptista, L. (2021). *Predict Students' Dropout and Academic Success*. UCI Machine Learning Repository. https://doi.org/10.24432/C5MC89
- [UCI ML Repository — Dataset #697](https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success)
- Scikit-learn MLPClassifier Documentation: https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html

---
