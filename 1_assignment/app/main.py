from fastapi import FastAPI, UploadFile, File, HTTPException
import pandas as pd
import pickle
import os
from model import train_best_model, load_model, predict_downtime

app = FastAPI()

DATA_PATH = "../data/uploaded_data.csv"
MODEL_PATH = "../models/best_model.pkl"

# Ensure directories exist
os.makedirs("data", exist_ok=True)
os.makedirs("models", exist_ok=True)

# Load model at startup
model = None


@app.post("/upload")
def upload_file(file: UploadFile = File(...)):
    try:
        df = pd.read_csv(file.file)
        df.to_csv(DATA_PATH, index=False)
        return {"message": "File uploaded successfully", "columns": df.columns.tolist()}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing file: {str(e)}")


@app.post("/train")
def train_model():
    if not os.path.exists(DATA_PATH):
        raise HTTPException(status_code=400, detail="No dataset found. Please upload a dataset first.")

    df = pd.read_csv(DATA_PATH)
    metrics = train_best_model(df)

    return {"message": "Model trained successfully", "metrics": metrics}


@app.post("/predict")
def predict(data: dict):
    global model
    if model is None:
        model = load_model()  # Load model only once

    prediction = predict_downtime(model, data)
    return prediction


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
