import uvicorn
import numpy as np
import tensorflow as tf

from fastapi import FastAPI, File, UploadFile
from io import BytesIO
from PIL import Image

MODEL = tf.keras.models.load_model('models/1')
CLASS_NAMES = ['Early Blight', 'Late Blight', 'Healthy']

# Building the app
app = FastAPI()


# API Endpoints

# Get method
@app.get("/ping")
async def ping():
    return "Hello TRoye"

# Post method
def read_file_as_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    return image

app.post("/predict")
async def predict(
    file : UploadFile = File(...)
):
    image = read_file_as_image(await file.read())
    img_batch = np.expand_dims(image, 0)
    predictions = MODEL.predict(img_batch)
    predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
    confidence = np.max(predictions[0])
    return {
        'class' : predicted_class,
        'confidence' : float(confidence)
    }

if __name__ == "__main__":
    uvicorn.run(app, host='localhost', port=8000)
