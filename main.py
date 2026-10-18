from fastapi import FastAPI, File, UploadFile

app = FastAPI()

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/inspect")
async def inspect_image(file: UploadFile = File(...)):
    data = await file.read()
    return {
        "filename": file.filename,
        "content_type": file.content_type,
        "size_bytes": len(data),
    }

