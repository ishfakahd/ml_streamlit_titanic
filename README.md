# Titanic Survival Prediction Web App

## Project Overview
This project involves the deployment of a **machine learning model using Streamlit**. It predicts passenger survival on the Titanic based on features such as age, sex, class, fare, and family details. The app allows interactive exploration, visualizations, and real-time predictions.

---

## Author
- Name: ITBIN-2211-0011
- Course: IT41043 - Intelligent Systems
- Year: 4th Year, BSc IT

---

## Dataset
- **Source:** Titanic Dataset from Kaggle
- **File:** `data/titanic.csv`
- **Description:** The dataset contains information about Titanic passengers, including `Survived` (target), `Pclass`, `Sex`, `Age`, `SibSp`, `Parch`, `Fare`, and `Embarked`.
- **Preprocessing Steps:**
  - Dropped irrelevant columns: `Name`, `Ticket`, `Cabin`
  - Handled missing values by dropping rows with NaN
  - Encoded categorical columns: `Sex` (0=male, 1=female), `Embarked` (one-hot encoding)

---

## Project Structure
ml_streamlit_titanic/

├── app.py # Streamlit application

├── requirements.txt # Python dependencies

├── model.pkl # Trained ML model

├── data/
│ └── Titanic-Dataset.csv # Dataset

├── notebooks/
│ └── model_training.ipynb # Model training notebook

└── README.md # Project documentation


---

## Features
- **Data Exploration:** View dataset overview, sample data, and filtering options.
- **Visualizations:** At least 3 charts including survival count and correlations.
- **Model Prediction:** Input passenger details via sidebar and get real-time survival prediction.
- **Model Performance:** Displays accuracy, confusion matrix, and evaluation metrics.

---

## Installation
1. Clone the repository:
```bash
git clone (https://github.com/ishfakahd/ml_streamlit_titanic.git)

2. Navigate to the project folder:

cd ml_streamlit_titanic


3. Install dependencies:

pip install -r requirements.txt


4. Run the Streamlit app:

streamlit run app.py

Deployment

Streamlit Cloud URL: Your Deployed App Link Here

All features work in the cloud environment, including interactive visualizations and model predictions.

Model Details

Algorithm: Logistic Regression

Features Used: Pclass, Sex, Age, SibSp, Parch, Fare, Embarked

Accuracy: XX% (replace with your trained model score)

Saved Model File: model.pkl

Reflection & Learning Outcomes

Gained hands-on experience in ML model development.

Learned how to preprocess real-world data and handle categorical features.

Built an interactive web application using Streamlit.

Learned the end-to-end ML deployment process including GitHub version control and Streamlit Cloud deployment.

