from contextlib import asynccontextmanager
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import numpy as np
import io
from keras.models import load_model
from PIL import Image
import base64
import uvicorn

# Initialize FastAPI with lifespan
app = FastAPI(title="MNIST Digit Classification API")


@app.get("/")
async def root():
    return {"message": "MNIST Classification API is running! Use /predict-file or /predict-draw endpoints."}

@app.post("/predict-file")
async def predict_file(file: UploadFile = File):
    """
    Make a prediction on an uploaded image file
    """
    # Read and preprocess the file
    model_path = "model/model/MNIST_keras_CNN.h5"
    try:
        model = load_model(model_path)
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"Model loading failed: {e}"})
    
    contents = await file.read()
    img = Image.open(io.BytesIO(contents))
    img = img.convert("L")  # Convert to grayscale
    img = img.resize((28, 28))
    img = np.array(img)
    img = img.reshape(1, 28, 28, 1)
    img = img.astype("float32") / 255

    # Make predictions
    predictions = model.predict(img)
    predicted_class = int(np.argmax(predictions))
    predicted_confidence = float(np.max(predictions))

    return JSONResponse(content={
        "predicted_class": predicted_class,
        "confidence": predicted_confidence
    })

@app.post("/predict-draw")
async def predict_draw(image_data: str = Form(...)):
    """
    Make a prediction on a drawn image (base64 encoded)
    """
    try:
        # Decode base64 image
        image_data = image_data.split(",")[1] if "," in image_data else image_data
        image_bytes = base64.b64decode(image_data)
        
        img = Image.open(io.BytesIO(image_bytes))
        img = img.convert("L")  # Convert to grayscale
        img = img.resize((28, 28))
        img = np.array(img)
        img = img.reshape(1, 28, 28, 1)
        img = img.astype("float32") / 255

        # Make predictions
        predictions = model.predict(img)
        predicted_class = int(np.argmax(predictions))
        predicted_confidence = float(np.max(predictions))

        return JSONResponse(content={
            "predicted_class": predicted_class,
            "confidence": predicted_confidence
        })
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": str(e)})

if __name__ == "__main__":
    uvicorn.run("app-fastapi:app", host="0.0.0.0", port=8000, reload=True)

