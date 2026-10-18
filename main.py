import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, Response, UploadFile
from inspector_stub import inspect as run_inspection

app = FastAPI()


async def load_image(file: UploadFile):
    data = await file.read()
    image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(status_code=400, detail="Could not read that image.")
    return image


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/inspect")
async def inspect_image(file: UploadFile = File(...)):
    image = await load_image(file)
    result = run_inspection(image)
    return {
        "filename": file.filename,
        "action": result["action"],
        "detections": result["detections"],
        "trace": result["trace"],
    }


@app.post("/inspect/image")
async def inspect_image_view(file: UploadFile = File(...)):
    image = await load_image(file)
    result = run_inspection(image)
    ok, buffer = cv2.imencode(".jpg", result["annotated_image"])
    return Response(content=buffer.tobytes(), media_type="image/jpeg")