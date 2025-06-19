# Diabetes-Prediction-using-Machine-Learning

## 📘 Overview
This project aims to develop classification models to predict whether an individual has diabetes based on diagnostic features from the Pima Indian Diabetes Dataset. The primary goal is to achieve the highest possible validation accuracy using various machine learning algorithms and hyperparameter tuning techniques.

---

## 🧠 Project Motivation
Diabetes is a chronic medical condition affecting millions worldwide. Early detection and diagnosis can significantly reduce complications. By leveraging machine learning, this project explores predictive modeling approaches to assist healthcare professionals in making informed decisions.

---

## 📂 Dataset
- **Source**: National Institute of Diabetes and Digestive and Kidney Diseases
- **Features**:
  - Pregnancies
  - Glucose
  - Blood Pressure
  - Skin Thickness
  - Insulin
  - BMI
  - Diabetes Pedigree Function
  - Age
  - Outcome (0 = Non-diabetic, 1 = Diabetic)

---

## 📊 Exploratory Data Analysis (EDA)
- Identified structural and statistical properties of the dataset.
- Replaced invalid zero entries with NaN and imputed with class-wise medians.
- Removed outliers using the **Local Outlier Factor (LOF)** technique.
- Standardized features using **RobustScaler** for better model performance.

---

## 🛠️ Model Building
Multiple classification algorithms were trained and evaluated:
- Logistic Regression
- K-Nearest Neighbors (KNN)
- Support Vector Machines (SVM)
- Decision Trees (CART)
- Random Forest
- XGBoost
- LightGBM

Each model was validated using **10-fold cross-validation** and compared using boxplots of accuracy scores.

---

## 🔧 Hyperparameter Tuning
Applied **RandomizedSearchCV** to optimize:
- Random Forest
- XGBoost
- LightGBM

The best-performing model was **XGBoost** with a cross-validation accuracy of approximately **90%**.

---

## 📈 Results
| Model           | Accuracy (CV) |
|----------------|----------------|
| Logistic Reg.  | ~85%           |
| KNN            | ~86%           |
| SVM            | ~87%           |
| Random Forest  | ~89%           |
| XGBoost        | **~90%**       |
| LightGBM       | ~89%           |

---

## 📝 Conclusion
- XGBoost performed best after tuning.
- Proper handling of missing values, outliers, and feature scaling significantly impacted model performance.
- The approach can be extended with deep learning models or feature engineering for further accuracy.

---
## 🖥️ How to Run
1. Clone the repository:
   ```bash
   git clone https://github.com/DipanjanaPal123/Diabetes-Prediction-using-Machine-Learning.git
   cd Diabetes Prediction using machine learning

## 📚 Requirements

- Python 
- NumPy
- Pandas
- Scikit-learn
- XGBoost
- LightGBM
- Matplotlib
- Seaborn
- Jupyter Notebook
