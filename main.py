import json
import time
import uuid
from aws_utils import save_run, send_alert
import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, Response, UploadFile
from agent import inspect as run_inspection
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
def home():
    return FileResponse("static/index.html")


async def load_image(file: UploadFile):
    data = await file.read()
    image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(status_code=400, detail="Could not read that image.")
    return image,data


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/inspect")
async def inspect_image(file: UploadFile = File(...)):
    image, data = await load_image(file)
    started = time.perf_counter()
    result = run_inspection(image)
    seconds = round(time.perf_counter() - started, 2)
    print(f"inspection took {seconds}s")

    run_id = uuid.uuid4().hex[:12]
    ok, buffer = cv2.imencode(".jpg", result["annotated_image"])
    record = {
        "run_id": run_id,
        "filename": file.filename,
        "action": result["action"],
        "status": result["status"],
        "summary": result["summary"],
        "detections": result["detections"],
        "trace": result["trace"],
    }
    saved = save_run(run_id, data, buffer.tobytes(), record)

    alerted = False
    if result["status"] == "POTENTIAL_PPE_ISSUES":
        alerted = send_alert(
            "Potential PPE issue: human review needed",
            f"Run {run_id}: {file.filename}\n{result['summary']['message']}\n\n"
            + json.dumps(result["potential_violations"], indent=2),
        )

    return {**record, "saved_to_s3": saved, "alert_sent": alerted}


@app.post("/inspect/image")
async def inspect_image_view(file: UploadFile = File(...)):
    image, data = await load_image(file)
    result = run_inspection(image)
    ok, buffer = cv2.imencode(".jpg", result["annotated_image"])
    return Response(content=buffer.tobytes(), media_type="image/jpeg")