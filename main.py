from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import numpy as np
from io import BytesIO
from PIL import Image
import uvicorn
import tensorflow as tf

app = FastAPI()

MODEL = tf.keras.models.load_model("../saved_models/1")

@app.get("/ping")
async def ping():
    return {"message": "Hello, I am alive"}  # Return a dictionary for JSON response

def read_file_as_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    return image


@app.post("/predict")
async def predict(file: UploadFile = File(...)): 
    image = read_file_as_image(await file.read())
    return


if __name__ == "__main__":
    uvicorn.run(app, host='localhost', port=8001)

