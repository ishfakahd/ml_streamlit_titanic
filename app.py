import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load dataset
df = sns.load_dataset('titanic')  # fixed the dataset name
st.title("Titanic Survival Prediction App")
st.write("Predict whether a passenger survived using ML model")

# Data Preprocessing
df = df[['survived', 'pclass', 'sex', 'age', 'fare']].dropna()
df['sex'] = df['sex'].map({'male': 0, 'female': 1})

# Show dataset info
st.write("### Dataset Overview")
st.write(df.describe())
st.write(df.head())

# Train/Test Split
X = df[['pclass', 'sex', 'age', 'fare']]
y = df['survived']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Model
model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
st.write(f"### Model Accuracy: {accuracy:.2f}")

# Model Performance
st.write("### Confusion Matrix")
fig, ax = plt.subplots()
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues', ax=ax)
st.pyplot(fig)

st.write("### Classification Report")
st.text(classification_report(y_test, y_pred))

# Data Visualization
st.write("### Survival Count")
fig2, ax2 = plt.subplots()
sns.countplot(x="survived", data=df, ax=ax2)
st.pyplot(fig2)

st.write("### Age Distribution")
fig3, ax3 = plt.subplots()
sns.histplot(df['age'], bins=20, ax=ax3)
st.pyplot(fig3)

# User Input for Prediction
st.sidebar.header("Enter Passenger Details")
pclass = st.sidebar.selectbox("Pclass", [1, 2, 3])
sex = st.sidebar.selectbox("Sex", ["male", "female"])
age = st.sidebar.slider("Age", 0, 80, 25)
fare = st.sidebar.slider("Fare", 0, 500, 50)

sex_numeric = 0 if sex == "male" else 1
input_data = pd.DataFrame([[pclass, sex_numeric, age, fare]], columns=['pclass','sex','age','fare'])
prediction = model.predict(input_data)[0]
st.write("### Prediction: ", "Survived ✅" if prediction == 1 else "Not Survived ❌")
