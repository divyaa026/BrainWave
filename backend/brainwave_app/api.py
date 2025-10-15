from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import numpy as np
import base64
import cv2
from io import BytesIO
from PIL import Image

try:
    # when executed as package
    from .models import DemoModels
except Exception:
    # fallback when running directly
    from models import DemoModels

app = FastAPI(title="BrainWave Analyzer API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# instantiate demo models once
models = DemoModels()


def image_to_data_url(img: np.ndarray) -> str:
    # img expected float [0,1]
    img_uint8 = (np.clip(img, 0.0, 1.0) * 255).astype('uint8')
    pil = Image.fromarray(img_uint8)
    buf = BytesIO()
    pil.save(buf, format='PNG')
    b64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    return f"data:image/png;base64,{b64}"


@app.post('/api/eeg-to-image')
async def eeg_to_image(payload: dict):
    # Expecting payload {"eeg": [..]}
    eeg = payload.get('eeg') if isinstance(payload, dict) else None
    if eeg is None:
        raise HTTPException(status_code=400, detail='Missing eeg field (array)')
    try:
        img = models.predict_image_from_eeg(np.array(eeg))
        data_url = image_to_data_url(img)
        return JSONResponse({"image_data_url": data_url})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post('/api/image-to-eeg')
async def image_to_eeg(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail='Could not decode image')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    try:
        eeg = models.predict_eeg_from_image(img)
        return JSONResponse({"eeg": eeg.tolist()})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    # Allow running `python api.py` to start the server for local dev
    import uvicorn
    uvicorn.run("api:app", host="127.0.0.1", port=8000, log_level="info")
