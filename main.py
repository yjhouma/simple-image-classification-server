from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse
from enum import Enum
import numpy as np
import io
import uvicorn
import cv2
import cvlib as cv
from cvlib.object_detection import draw_bbox


class ModelSelector(str, Enum):
    yolov3tiny = 'yolov3-tiny'
    yolov3 = 'yolov3'


app = FastAPI(debug=True, title="Simple ML Server")



@app.get("/")
def home():
    return "The Server is Running OK!"

@app.post("/predict")
def predict(model: ModelSelector, file: UploadFile=File(...)):
    
    # Input validation
    filename = file.filename
    fileExtension = filename.split(".")[-1] in ("jpg", "jpeg", "png")
    if not fileExtension:
        raise HTTPException(status_code=415, detail="Unsupported file provided.")
    
    #Transform file to CV2 image (numpy array)
    image_stream = io.BytesIO(file.file.read())

    image_stream.seek(0)

    file_bytes = np.asarray(bytearray(image_stream.read()), dtype=np.uint8)

    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)


    # Detect Object
    bbox, label, conf = cv.detect_common_objects(image, model=model)

    output_image = draw_bbox(image, bbox, label, conf)

    cv2.imwrite(f'data/uploaded_images/{filename}', output_image)

    # Send back Prediction

    file_image = open(f'data/uploaded_images/{filename}', mode="rb")

    return StreamingResponse(file_image,media_type="image/jpeg")




