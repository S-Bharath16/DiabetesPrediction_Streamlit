import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Load data
data = pd.read_csv('diabetes.csv')
x = data.drop("Outcome", axis=1)
y = data['Outcome']

# Train model
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
model = LogisticRegression()
model.fit(x_train, y_train)

# Streamlit UI
st.title("Diabetes Prediction App")

st.write("### Enter the required values for prediction:")

# User input for prediction
val1 = st.number_input("Pregnancies", min_value=0.0, max_value=20.0)
val2 = st.number_input("Glucose", min_value=0.0, max_value=200.0)
val3 = st.number_input("Blood Pressure", min_value=0.0, max_value=122.0)
val4 = st.number_input("Skin Thickness", min_value=0.0, max_value=100.0)
val5 = st.number_input("Insulin", min_value=0.0, max_value=900.0)
val6 = st.number_input("BMI", min_value=0.0, max_value=70.0)
val7 = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=3.0)
val8 = st.number_input("Age", min_value=0.0, max_value=120.0)

# Prediction button
if st.button("Predict"):
    pred = model.predict([[val1, val2, val3, val4, val5, val6, val7, val8]])
    result = "Positive" if pred[0] == 1 else "Negative"
    st.write("### Prediction Result:", result)
