from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import numpy as np
import base64
import cv2
import time
import traceback
from io import BytesIO
from PIL import Image
import logging

try:
    # when executed as package
    from .models import DemoModels
except Exception:
    # fallback when running directly
    from models import DemoModels

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="BrainWave Analyzer API",
    description="Advanced multi-modal deep learning API for EEG-Image synthesis",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global models instance
models = None

class EEGRequest(BaseModel):
    """EEG input request model"""
    eeg: List[float] = Field(..., description="EEG signal array")
    brain_state: Optional[str] = Field(None, description="Brain state hint: relaxed, focused, active, motor, sleep")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")

class EEGResponse(BaseModel):
    """EEG to Image response model"""
    success: bool
    image_data_url: str
    confidence_score: float
    processing_time_ms: float
    model_info: Dict[str, Any]
    metadata: Dict[str, Any]

class ImageResponse(BaseModel):
    """Image to EEG response model"""
    success: bool
    eeg: List[float]
    confidence_score: float
    processing_time_ms: float
    model_info: Dict[str, Any]
    metadata: Dict[str, Any]

class HealthResponse(BaseModel):
    """Health check response model"""
    status: str
    models_loaded: bool
    model_info: Dict[str, Any]
    uptime_seconds: float

@app.on_event("startup")
async def startup_event():
    """Initialize models on startup"""
    global models
    try:
        logger.info("Initializing BrainWave models...")
        models = DemoModels()
        logger.info("Models initialized successfully!")
    except Exception as e:
        logger.error(f"Failed to initialize models: {e}")
        raise

def image_to_data_url(img: np.ndarray) -> str:
    """Convert numpy image array to data URL"""
    try:
        # Ensure image is in correct range [0,1]
    img_uint8 = (np.clip(img, 0.0, 1.0) * 255).astype('uint8')
        
        # Convert to PIL Image
    pil = Image.fromarray(img_uint8)
        
        # Convert to base64
    buf = BytesIO()
        pil.save(buf, format='PNG', optimize=True)
    b64 = base64.b64encode(buf.getvalue()).decode('utf-8')
        
    return f"data:image/png;base64,{b64}"
    except Exception as e:
        logger.error(f"Error converting image to data URL: {e}")
        raise

def validate_eeg_input(eeg_data: List[float]) -> np.ndarray:
    """Validate and preprocess EEG input"""
    try:
        # Convert to numpy array
        eeg_array = np.array(eeg_data, dtype=np.float32)
        
        # Basic validation
        if len(eeg_array) == 0:
            raise ValueError("EEG data cannot be empty")
        
        if np.any(np.isnan(eeg_array)) or np.any(np.isinf(eeg_array)):
            raise ValueError("EEG data contains invalid values (NaN or Inf)")
        
        # Check for reasonable range (rough sanity check)
        if np.std(eeg_array) > 1000:  # Arbitrary threshold
            logger.warning("EEG data has very high variance, may contain artifacts")
        
        return eeg_array
        
    except Exception as e:
        logger.error(f"EEG validation error: {e}")
        raise ValueError(f"Invalid EEG data: {str(e)}")

def validate_image_input(file: UploadFile) -> np.ndarray:
    """Validate and preprocess image input"""
    try:
        # Check file size (limit to 10MB)
        contents = file.file.read()
        if len(contents) > 10 * 1024 * 1024:
            raise ValueError("Image file too large (max 10MB)")
        
        # Reset file pointer
        file.file.seek(0)
        
        # Decode image
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            raise ValueError("Could not decode image file")
        
        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        
        # Basic validation
        if img.shape[0] < 32 or img.shape[1] < 32:
            raise ValueError("Image too small (minimum 32x32 pixels)")
        
        return img
        
    except Exception as e:
        logger.error(f"Image validation error: {e}")
        raise ValueError(f"Invalid image data: {str(e)}")

@app.get("/", response_model=HealthResponse)
async def root():
    """Root endpoint with API information"""
    global models
    uptime = time.time() - app.state.start_time if hasattr(app.state, 'start_time') else 0
    
    return HealthResponse(
        status="healthy" if models is not None else "models_not_loaded",
        models_loaded=models is not None,
        model_info=models.get_model_info() if models else {},
        uptime_seconds=uptime
    )

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    global models
    uptime = time.time() - app.state.start_time if hasattr(app.state, 'start_time') else 0
    
    return HealthResponse(
        status="healthy" if models is not None else "models_not_loaded",
        models_loaded=models is not None,
        model_info=models.get_model_info() if models else {},
        uptime_seconds=uptime
    )

@app.get("/api/model-info")
async def get_model_info():
    """Get detailed model information"""
    global models
    if models is None:
        raise HTTPException(status_code=503, detail="Models not loaded")
    
    return models.get_model_info()

@app.post("/api/eeg-to-image", response_model=EEGResponse)
async def eeg_to_image(request: EEGRequest):
    """Convert EEG signal to image"""
    global models
    if models is None:
        raise HTTPException(status_code=503, detail="Models not loaded")
    
    start_time = time.time()
    
    try:
        # Validate input
        eeg_array = validate_eeg_input(request.eeg)
        
        # Generate image
        img = models.predict_image_from_eeg(eeg_array)
        
        # Convert to data URL
        data_url = image_to_data_url(img)
        
        # Calculate processing time
        processing_time = (time.time() - start_time) * 1000  # Convert to ms
        
        # Get confidence score
        confidence = models.get_confidence_score('eeg_to_image')
        
        # Prepare response
        response = EEGResponse(
            success=True,
            image_data_url=data_url,
            confidence_score=confidence,
            processing_time_ms=processing_time,
            model_info=models.get_model_info(),
            metadata={
                "input_length": len(request.eeg),
                "brain_state_hint": request.brain_state,
                "output_shape": img.shape,
                "timestamp": time.time()
            }
        )
        
        logger.info(f"EEG to Image conversion completed in {processing_time:.2f}ms")
        return response
        
    except ValueError as e:
        logger.warning(f"Validation error in EEG to Image: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error in EEG to Image conversion: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail="Internal server error during EEG to Image conversion")

@app.post("/api/image-to-eeg", response_model=ImageResponse)
async def image_to_eeg(file: UploadFile = File(...)):
    """Convert image to EEG signal"""
    global models
    if models is None:
        raise HTTPException(status_code=503, detail="Models not loaded")
    
    start_time = time.time()
    
    try:
        # Validate image
        img = validate_image_input(file)
        
        # Generate EEG
        eeg = models.predict_eeg_from_image(img)
        
        # Calculate processing time
        processing_time = (time.time() - start_time) * 1000  # Convert to ms
        
        # Get confidence score
        confidence = models.get_confidence_score('image_to_eeg')
        
        # Prepare response
        response = ImageResponse(
            success=True,
            eeg=eeg.tolist(),
            confidence_score=confidence,
            processing_time_ms=processing_time,
            model_info=models.get_model_info(),
            metadata={
                "input_shape": img.shape,
                "filename": file.filename,
                "content_type": file.content_type,
                "output_length": len(eeg),
                "timestamp": time.time()
            }
        )
        
        logger.info(f"Image to EEG conversion completed in {processing_time:.2f}ms")
        return response
        
    except ValueError as e:
        logger.warning(f"Validation error in Image to EEG: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error in Image to EEG conversion: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail="Internal server error during Image to EEG conversion")

@app.post("/api/generate-sample-eeg")
async def generate_sample_eeg(brain_state: str = "relaxed"):
    """Generate sample EEG signal for demonstration"""
    global models
    if models is None:
        raise HTTPException(status_code=503, detail="Models not loaded")
    
    try:
        # Validate brain state
        valid_states = ["relaxed", "focused", "active", "motor", "sleep"]
        if brain_state.lower() not in valid_states:
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid brain state. Must be one of: {valid_states}"
            )
        
        # Generate sample EEG
        eeg = models.generate_sample_eeg(brain_state.lower())
        
        return JSONResponse({
            "success": True,
            "eeg": eeg.tolist(),
            "brain_state": brain_state.lower(),
            "length": len(eeg),
            "metadata": {
                "description": f"Sample EEG signal for {brain_state} state",
                "timestamp": time.time()
            }
        })
        
    except Exception as e:
        logger.error(f"Error generating sample EEG: {e}")
        raise HTTPException(status_code=500, detail="Internal server error during sample EEG generation")

@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    """Add processing time to response headers"""
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response

@app.on_event("startup")
async def set_start_time():
    """Set application start time"""
    app.state.start_time = time.time()


if __name__ == "__main__":
    # Allow running `python api.py` to start the server for local dev
    import uvicorn
    uvicorn.run("api:app", host="127.0.0.1", port=8000, log_level="info")
