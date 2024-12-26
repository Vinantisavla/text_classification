from fastapi import APIRouter, HTTPException
from model.predict import ModelPredictor
import os
router = APIRouter()
predictor = ModelPredictor("model/svm_model.pkl")


@router.post("/predict/")
def predict(text: str):
    try:
        result = predictor.predict(text)
        return {"status": "success", "data": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

print("Current Working Directory:", os.getcwd())
print("Model file exists:", os.path.exists("model/svm_model.pkl"))