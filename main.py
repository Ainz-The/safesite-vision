import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from inspector_stub import inspect as run_inspection

app = FastAPI()

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/inspect")
async def inspect_image(file: UploadFile = File(...)):
    data = await file.read()
    image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(status_code=400, detail="Could not read that image.")
    result = run_inspection(image)
    return {
        "filename": file.filename,
        "action": result["action"],
        "detections": result["detections"],
        "trace": result["trace"],
    }