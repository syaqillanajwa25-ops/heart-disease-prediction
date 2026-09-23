# Heart Disease Prediction using Machine Learning

A machine learning project for predicting the presence of heart disease based on patient clinical and health-related attributes.

The project covers exploratory data analysis, data preprocessing, model training, hyperparameter tuning, and model evaluation using Logistic Regression and Random Forest.

## Project Overview

The objective of this project is to build a classification model that can predict whether a patient shows signs of heart disease based on several medical attributes.

The dataset contains **303 observations and 14 columns**, consisting of 13 predictor variables and one target variable.

Some of the features include:

- Age
- Sex
- Chest pain type
- Resting blood pressure
- Cholesterol
- Fasting blood sugar
- Resting ECG results
- Maximum heart rate
- Exercise-induced angina
- ST depression
- Number of major vessels
- Thalassemia

## Tech Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Jupyter Notebook
- Logistic Regression
- Random Forest
- GridSearchCV

## Machine Learning Workflow

The project follows a structured machine learning pipeline:

1. Data ingestion
2. Exploratory Data Analysis (EDA)
3. Data preprocessing
4. Train-test splitting
5. Logistic Regression training
6. Random Forest training
7. Hyperparameter tuning using GridSearchCV
8. Model evaluation
9. Selection of the best-performing model

## Models

### Logistic Regression

Logistic Regression was used as a baseline classification model to evaluate the relationship between patient characteristics and heart disease prediction.

### Random Forest

Random Forest was used as an ensemble-based classification model capable of capturing more complex relationships between features.

The Random Forest model was further optimized using **GridSearchCV** to identify the best hyperparameter configuration.

## Model Evaluation

The models were evaluated using classification metrics such as:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC

The tuned Random Forest model achieved approximately:

- **Accuracy: 81.97%**
- **ROC-AUC: 0.912**
- **Best Cross-Validation ROC-AUC: 0.922**

These results showed that the tuned Random Forest model provided strong classification performance on the dataset.

## Project Structure

```text
heart-disease-prediction/
│
├── ExplorasiHeartAttack.ipynb
├── data_ingestion.py
├── pre_processing.py
├── train.py
├── evaluation.py
├── pipeline.py
├── requirements.txt
└── README.md
