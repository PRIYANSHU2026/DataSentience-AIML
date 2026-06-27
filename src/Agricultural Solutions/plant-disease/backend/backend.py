import os
import json
import numpy as np
import tensorflow as tf
import tf_keras 
from fastapi import FastAPI, File, UploadFile, HTTPException
from PIL import Image
import io

app = FastAPI(title="Plant Disease Detection API")

# Path update 
MODEL_PATH = r"D:\Plant_Diseases\Plant_Disease\Final_model\final_model_files\fine_tuned_model.h5"
JSON_PATH = r"D:\Plant_Diseases\Plant_Disease\Final_model\final_model_files\class_indices.json"

model = None
class_indices = None

@app.on_event("startup")
def load_model_and_indices():
    global model, class_indices
    # Model Loading
    if os.path.exists(MODEL_PATH):
        try:
       
            model = tf_keras.models.load_model(MODEL_PATH, compile=False)
            print("SUCCESS: Fine-tuned model loaded!")
        except Exception as e:
            print(f"Model Load Error: {e}")
    

    if os.path.exists(JSON_PATH):
        with open(JSON_PATH, 'r') as f:
            class_indices = json.load(f)
            print("Class Indices loaded successfully!")

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    global model, class_indices
    if model is None:
        raise HTTPException(status_code=500, detail="Model not initialized.")

    try:

        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert('RGB')
        
        # 1. Resize: Training mein 224x224 use kiya tha
        image = image.resize((224, 224)) 
        
        # 2. Preprocess
        img_array = tf_keras.preprocessing.image.img_to_array(image)
        img_array = np.expand_dims(img_array, axis=0)
        
        # 3. Scaling: ImageDataGenerator(rescale=1./255) ke liye zaruri
        img_array = img_array / 255.0 

        # 4. Predict
        predictions = model.predict(img_array)
        
        # 5. Result Extraction
        predicted_index = str(np.argmax(predictions[0]))
        confidence = float(np.max(predictions[0]) * 100)
        
        predicted_label = class_indices.get(predicted_index, "Unknown Disease")

        return {
            "class": predicted_label,
            "confidence": round(confidence, 2),
            "status": "Success"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='127.0.0.1', port=8000)