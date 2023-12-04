import cv2
import uvicorn
import numpy as np

from fastapi import FastAPI, UploadFile
from fastapi.responses import FileResponse
from io import BytesIO
from starlette.responses import StreamingResponse

from worker import Worker

app = FastAPI()


@app.get("/")
def index():
    return {"message": "Yes, I am still living."}


@app.post("/upload")
async def upload_file(file: UploadFile) -> StreamingResponse:
    # read upload data
    img_data = await file.read()

    # decode img data
    img = cv2.imdecode(np.frombuffer(img_data, np.uint8), cv2.IMREAD_COLOR)

    # build worker to process deep learning algorithms
    worker = Worker(img)
    channel_down, channel_up = worker.yolo_detect()

    channel_down_class, channel_up_class = worker.predict(channel_down, channel_up)

    return {"down": channel_down_class, "up": channel_up_class}


if __name__ == "__main__":
    uvicorn.run(app, port=8000, host="0.0.0.0")
