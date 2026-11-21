# 🎓 Student Dropout Prediction

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Mohamedarshathsaleem/Student-Dropout-Prediction/blob/main/studentoutcomes_mlp.ipynb)
![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-MLPClassifier-F7931E?logo=scikitlearn&logoColor=white)

**A lightweight neural-network classifier that predicts whether a student will *Drop out*, stay *Enrolled* or *Graduate*, using only five facts known at enrolment, plus an interactive prompt to score a new student.**

---

## Why five features?

The UCI dataset has 36 features, but many of them (semester grades, approved units) only exist *after* a student has started studying. I deliberately used only information an admissions office has on **day one**, so the model could flag at-risk students before the first semester:

| Feature | Type | Preprocessing |
|---|---|---|
| Gender | binary (1 = male, 0 = female) | as is |
| Marital status | categorical code 1–6 | as is |
| Admission grade | continuous (0–200) | `StandardScaler` |
| Age at enrollment | continuous | `StandardScaler` |
| Scholarship holder | binary | as is |

**Target:** `Dropout` / `Enrolled` / `Graduate`, label-encoded.

## 🧠 Model

- `MLPClassifier(hidden_layer_sizes=(16,), max_iter=300, random_state=42)`: one hidden layer of 16 ReLU units, Adam optimiser
- 80/20 train/test split (`random_state=42`)
- Evaluated with scikit-learn's `classification_report` (precision, recall and F1 per class)
- The final cell asks for the five inputs, applies the same scaler and prints the predicted outcome

## 📈 Results

<!-- TODO(Arshath): run the notebook in Colab and paste the classification_report output here,
     e.g. overall accuracy and the per-class F1 scores. A real number is far more convincing than none. -->

| Class | Precision | Recall | F1 |
|---|---|---|---|
| Dropout | – | – | – |
| Enrolled | – | – | – |
| Graduate | – | – | – |
| **Accuracy** | | | **–** |

## 📊 Dataset

[Predict Students' Dropout and Academic Success](https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success) (UCI ML Repository, ID 697, CC BY 4.0): 4,424 undergraduates from a Portuguese higher-education institution. The classes are imbalanced, with Graduate the largest.

The notebook reads a CSV export of this dataset published from Google Sheets. To use the official source instead:

```python
from ucimlrepo import fetch_ucirepo   # pip install ucimlrepo
data = fetch_ucirepo(id=697)
df = data.data.features.join(data.data.targets)
```

## 🚀 Running it

**Easiest:** click **Open in Colab** above and choose *Runtime → Run all*.

**Locally:**

```bash
git clone https://github.com/Mohamedarshathsaleem/Student-Dropout-Prediction.git
cd Student-Dropout-Prediction
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook studentoutcomes_mlp.ipynb
```

## 🧭 Next steps

- Handle class imbalance (`class_weight` on a tree model, or SMOTE) and report macro-F1, which matters most for the small *Enrolled* class
- Compare against logistic regression and gradient boosting baselines
- Add a second model that uses first-semester results, to show how much accuracy improves after one semester
- Wrap the predictor in a small Streamlit or FastAPI app with a live demo

## 📚 References

- Realinho, V., Machado, J., Baptista, L., & Martins, M. V. (2022). *Predicting Student Dropout and Academic Success.* Data, 7(11), 146. https://doi.org/10.3390/data7110146
- [scikit-learn MLPClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html)

## 📄 License

Code: [MIT](LICENSE). Dataset: CC BY 4.0 (UCI).
