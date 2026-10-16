import streamlit as st
import joblib
import pandas as pd

# Load trained model
bundle = joblib.load("decision_tree_bundle.pkl")
model = bundle["model"]
feature_names = bundle["feature_names"]
target_names = bundle["target_names"]

st.title("🌸 Iris Flower Prediction")
st.write("Decision Tree Machine Learning Model")

st.subheader("Enter Flower Measurements")

sepal_length = st.number_input(
    "Sepal length (cm)", 4.3, 7.9, 5.1
)
sepal_width = st.number_input(
    "Sepal width (cm)", 2.0, 4.4, 3.5
)
petal_length = st.number_input(
    "Petal length (cm)", 1.0, 6.9, 1.4
)
petal_width = st.number_input(
    "Petal width (cm)", 0.1, 2.5, 0.2
)

if st.button("Predict Flower"):
    inputs = pd.DataFrame(
        [[sepal_length, sepal_width,
          petal_length, petal_width]],
        columns=feature_names
    )

    prediction = model.predict(inputs)[0]
    st.success(
        f"Predicted Flower: {target_names[prediction]}"
)
