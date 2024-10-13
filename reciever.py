from fastapi import FastAPI, File, UploadFile
import cv2
import numpy as np
from fastapi.responses import JSONResponse
from typing import List


app = FastAPI()


@app.post("/upload/")
async def upload_images(files: List[UploadFile] = File(...)):
    results = []

    for file in files:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is not None:
            height, width = img.shape[:2]
            results.append({"filename": file.filename, "width": width, "height": height})
        else:
            results.append({"filename": file.filename, "error": "Invalid image"})

    return JSONResponse(results)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8010)
