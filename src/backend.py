from fastapi import FastAPI, Request
import uvicorn
import pickle
import numpy as np

app = FastAPI()


@app.get("/")
def read_root():
    result = {
        "status": 200,
        "message": "Hooreee FastAPI Jalaannn..." 
    }
    return result

# check model loading
@app.get('/check-model')
def check_model():
    try:
        # try to load model
        with open('models/dtc_model.pkl', 'rb') as file:
            model = pickle.load(file)       
        result = {
            "status": 200,
            "message": "Model loaded successfully",
        }
        return result
    except Exception as e:
        result = {
            "status": 500,
            "message": "Model loading failed",
            "error": str(e)
        }
        return result

# Predict
@app.post('/predict')
async def predict(request: Request):
    # get data from request
    data = await request.json()

    age = data["Age"]
    pregnancies = data["Pregnancies"]
    glucose = data["Glucose"]
    blood_pressure = data["BloodPressure"]
    skin_thickness = data["SkinThickness"]
    insulin = data["Insulin"]
    bmi = data["BMI"]
    diabetes_pedigree_function = data["DiabetesPedigreeFunction"]

    # load model
    with open('models/production_model.pkl', 'rb') as file:
        model = pickle.load(file)
        
    # label 
    label = ['Anda Tidak Beresiko Diabetes', 'Anda Beresiko Diabetes']

    # validation input
    if age < 0 or pregnancies < 0 or glucose < 0 or blood_pressure < 0 or skin_thickness < 0 or insulin < 0 or bmi < 0.0 or diabetes_pedigree_function < -2.00 :
        result = {
            "status": 400,
            "message": "Input value cannot be negative"
        }
        return result
    
    # prediction
    try:
        prediction = model.predict(np.array(data).reshape(1, -1))
        result = {
            "status": 200,
            "message": "Prediction success",
            "prediction": str(prediction[0])
        }
        return result
    except Exception as e:
        result = {
            "status": 500,
            "message": "Prediction failed",
            "error": str(e)
        }
        return result
    
# Run API with uvicorn
if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8000)