import streamlit as st
import requests
import joblib
from PIL import Image

# Load and set images in the first place
header_images = Image.open('assets/banner.png')
st.image(header_images)

# Add some information about the service
st.title("Diabetes Detection")
st.subheader("Just enter variabel below then click Predict button :sunglasses:")

# Create form of input
with st.form(key = "data_form"):
    # Create box for number input
    age = st.number_input(
        label = "1.\tEnter Age Value:",
        min_value = 0,
        max_value = 100,
        help = "Value range from 0 to 100"
    )

    pregnancies = st.number_input(
        label = "2.\tEnter Pregnancies Value:",
        min_value = 0,
        max_value = 20,
        help = "Value range from 0 to 20"
    )
    
    glucose = st.number_input(
        label = "3.\tEnter Glucose Value:",
        min_value = 0,
        max_value = 250,
        help = "Value range from 0 to 250"
    )

    blood_pressure = st.number_input(
        label = "4.\tEnter Blood Pressure Value:",
        min_value = 0,
        max_value = 200,
        help = "Value range from 0 to 200"
    )

    skin_thickness= st.number_input(
        label = "5.\tEnter Skin Thickness Value:",
        min_value = 0,
        max_value = 100,
        help = "Value range from 0 to 100"
    )

    insulin = st.number_input(
        label = "6.\tEnter Insulin Value:",
        min_value = 0,
        max_value = 900,
        help = "Value range from 0 to 900"
    )

    bmi = st.number_input(
        label = "7.\tEnter BMI Value:",
        min_value = 0,
        max_value = 100,
        help = "Value range from 0 to 100"
    )

    diabetes_pedigree_function = st.number_input(
        label = "8.\tEnter Raw Ethanol Value:",
        min_value = -2,
        max_value = 4,
        help = "Value range from -2 to 4"
    )
    
    # Create button to submit the form
    submitted = st.form_submit_button("Predict")

    # Condition when form submitted
    if submitted:
        # Create dict of all data in the form
        raw_data = {
            "Age": age,
            "Pregnancies": pregnancies,
            "Glucose": glucose,
            "BloodPressure": blood_pressure,
            "SkinThickness": skin_thickness,
            "Insulin": insulin,
            "BMI": bmi,
            "DiabetesPedigreeFunction": diabetes_pedigree_function
        }

        # Create loading animation while predicting
        with st.spinner("Sending data to prediction server ..."):
            res = requests.post("http://localhost:8080/predict", json = raw_data).json()
            
        # Parse the prediction result
        if res["error_msg"] != "":
            st.error("Error Occurs While Predicting: {}".format(res["error_msg"]))
        else:
            if res["res"] != "Tidak ada api.":
                st.warning("Ada api.")
            else:
                st.success("Tidak ada api.")