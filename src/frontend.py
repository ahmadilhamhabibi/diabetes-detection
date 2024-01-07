import streamlit as st
import requests


st.image('assets/banner.png', caption='Diabetes Detection')
st.title('Diabetes Prediction')
st.markdown('Created by: Ahmad Ilham Habibi | Batch Period: NOV 2023 | HAPPY NEW YEAR')

st.subheader('Just enter variabel below then click Predict button :sunglasses:')

# Form Input
with st.form("diabetes-form"):
    age = st.number_input('Age', min_value=0,  step=1, max_value=100,  
                          help='Value range from 0 to 100')
    pregnancies = st.number_input('Pregnancies', min_value=0,  step=1, max_value=20,
                          help='Value range from 0 to 20')
    glucose = st.number_input('Glucose', min_value=0,  step=1, max_value=250,
                          help='Value range from 0 to 250')
    blood_pressure = st.number_input('Blood Pressure', min_value=0,  step=1, max_value=200,  
                          help='Value range from 0 to 200')
    skin_thickness = st.number_input('Skin Thickness', min_value=0,  step=1, max_value=100,  
                          help='Value range from 0 to 100')
    insulin = st.number_input('Insulin', min_value=0,  step=1, max_value=900,  
                          help='Value range from 0 to 900')
    bmi = st.number_input('BMI', min_value=0.0,  step=0.1, max_value=100.0,  
                          help='Value range from 0 to 100')
    diabetes_pedigree_function = st.number_input('Diabetes Pedigree Function', min_value=-2.000,  step=0.001, max_value=4.000,
                          help='Value range from -2 to 4')
    
    submit_button = st.form_submit_button(label='predict')
    
    if submit_button:
        data = {
            "Age": age,
            "Pregnancies": pregnancies,
            "Glucose": glucose,
            "BloodPressure": blood_pressure,
            "SkinThickness": skin_thickness,
            "Insulin": insulin,
            "BMI": bmi,
            "DiabetesPedigreeFunction": diabetes_pedigree_function
        }
            
        
        with st.spinner('Predicting...'):
            
            # Send request to backend
            response = requests.post('http://backend:8000/predict', json=data)
            result = response.json()
            
            # if success
            if result['status'] == 200:
                st.success(f"Prediction Success,  **{result['prediction']}** ")
                st.balloons()
                
            else:
                st.error('Prediction Failed')
        
    
    