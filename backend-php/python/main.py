"""
GestPay FacePay FastAPI Service
Handles face encoding, verification, and ML model logic for facial recognition payments
"""

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import face_recognition
import numpy as np
import cv2
from PIL import Image
import io
import base64
import json
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="GestPay FacePay API",
    description="Face recognition service for GestPay payment system",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models
class FaceEmbedding(BaseModel):
    user_id: int
    embedding: List[float]

class VerificationRequest(BaseModel):
    image_base64: str
    stored_embeddings: List[FaceEmbedding]
    confidence_threshold: Optional[float] = 0.6

class RegistrationRequest(BaseModel):
    image_base64: str

class VerificationResponse(BaseModel):
    match: bool
    user_id: Optional[int] = None
    confidence: float
    message: str

class RegistrationResponse(BaseModel):
    success: bool
    embedding: Optional[List[float]] = None
    message: str

# Utility functions
def decode_base64_image(base64_string: str) -> np.ndarray:
    """Decode base64 image string to numpy array"""
    try:
        # Remove data URL prefix if present
        if base64_string.startswith('data:image'):
            base64_string = base64_string.split(',')[1]
        
        # Decode base64
        image_data = base64.b64decode(base64_string)
        
        # Convert to PIL Image
        pil_image = Image.open(io.BytesIO(image_data))
        
        # Convert to RGB if necessary
        if pil_image.mode != 'RGB':
            pil_image = pil_image.convert('RGB')
        
        # Convert to numpy array
        image_array = np.array(pil_image)
        
        return image_array
    except Exception as e:
        logger.error(f"Error decoding base64 image: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Invalid image format: {str(e)}")

def extract_face_encoding(image: np.ndarray) -> Optional[np.ndarray]:
    """Extract face encoding from image"""
    try:
        # Find face locations
        face_locations = face_recognition.face_locations(image, model="hog")
        
        if len(face_locations) == 0:
            return None
        
        # Use the first (largest) face found
        face_encodings = face_recognition.face_encodings(image, face_locations)
        
        if len(face_encodings) == 0:
            return None
        
        return face_encodings[0]
    except Exception as e:
        logger.error(f"Error extracting face encoding: {str(e)}")
        return None

def calculate_face_distance(encoding1: np.ndarray, encoding2: np.ndarray) -> float:
    """Calculate distance between two face encodings"""
    return float(face_recognition.face_distance([encoding1], encoding2)[0])

def is_face_match(encoding1: np.ndarray, encoding2: np.ndarray, threshold: float = 0.6) -> bool:
    """Check if two face encodings match within threshold"""
    distance = calculate_face_distance(encoding1, encoding2)
    return distance <= threshold

# API Endpoints
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "GestPay FacePay API",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0"
    }

@app.post("/register-face", response_model=RegistrationResponse)
async def register_face(request: RegistrationRequest):
    """
    Extract face encoding from uploaded image for registration
    """
    try:
        logger.info("Processing face registration request")
        
        # Decode image
        image = decode_base64_image(request.image_base64)
        
        # Extract face encoding
        encoding = extract_face_encoding(image)
        
        if encoding is None:
            return RegistrationResponse(
                success=False,
                message="No clear face detected in the image. Please ensure your face is clearly visible and well-lit."
            )
        
        # Convert to list for JSON serialization
        embedding_list = encoding.tolist()
        
        logger.info("Face encoding extracted successfully")
        
        return RegistrationResponse(
            success=True,
            embedding=embedding_list,
            message="Face registered successfully"
        )
        
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Error in face registration: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Face registration failed: {str(e)}")

@app.post("/verify-face", response_model=VerificationResponse)
async def verify_face(request: VerificationRequest):
    """
    Verify uploaded face against stored embeddings
    """
    try:
        logger.info(f"Processing face verification request with {len(request.stored_embeddings)} stored faces")
        
        if not request.stored_embeddings:
            return VerificationResponse(
                match=False,
                confidence=0.0,
                message="No stored face embeddings provided for comparison"
            )
        
        # Decode image
        image = decode_base64_image(request.image_base64)
        
        # Extract face encoding from uploaded image
        uploaded_encoding = extract_face_encoding(image)
        
        if uploaded_encoding is None:
            return VerificationResponse(
                match=False,
                confidence=0.0,
                message="No clear face detected in the uploaded image. Please ensure your face is clearly visible."
            )
        
        # Compare with stored embeddings
        best_match_user_id = None
        best_confidence = 0.0
        best_distance = float('inf')
        
        for stored_face in request.stored_embeddings:
            try:
                # Convert stored embedding back to numpy array
                stored_encoding = np.array(stored_face.embedding)
                
                # Calculate distance (lower is better)
                distance = calculate_face_distance(uploaded_encoding, stored_encoding)
                
                # Convert distance to confidence (higher is better)
                confidence = max(0.0, 1.0 - distance)
                
                logger.info(f"User {stored_face.user_id}: distance={distance:.3f}, confidence={confidence:.3f}")
                
                # Check if this is the best match so far
                if distance < best_distance:
                    best_distance = distance
                    best_confidence = confidence
                    best_match_user_id = stored_face.user_id
                    
            except Exception as e:
                logger.error(f"Error comparing with user {stored_face.user_id}: {str(e)}")
                continue
        
        # Determine if match meets threshold
        is_match = best_distance <= request.confidence_threshold
        
        if is_match and best_match_user_id:
            logger.info(f"Face match found: User {best_match_user_id} with confidence {best_confidence:.3f}")
            return VerificationResponse(
                match=True,
                user_id=best_match_user_id,
                confidence=round(best_confidence, 3),
                message=f"Face verified successfully for user {best_match_user_id}"
            )
        else:
            logger.info(f"No face match found. Best confidence: {best_confidence:.3f}, threshold: {request.confidence_threshold}")
            return VerificationResponse(
                match=False,
                confidence=round(best_confidence, 3),
                message="Face verification failed. No matching face found or confidence too low."
            )
            
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Error in face verification: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Face verification failed: {str(e)}")

@app.post("/batch-verify")
async def batch_verify_faces(
    images: List[UploadFile] = File(...),
    stored_embeddings: str = Form(...)
):
    """
    Verify multiple face images at once (for testing purposes)
    """
    try:
        # Parse stored embeddings
        embeddings_data = json.loads(stored_embeddings)
        stored_faces = [FaceEmbedding(**face) for face in embeddings_data]
        
        results = []
        
        for image_file in images:
            # Read image file
            image_data = await image_file.read()
            
            # Convert to base64
            image_base64 = base64.b64encode(image_data).decode('utf-8')
            
            # Create verification request
            verify_request = VerificationRequest(
                image_base64=image_base64,
                stored_embeddings=stored_faces
            )
            
            # Verify face
            result = await verify_face(verify_request)
            results.append({
                "filename": image_file.filename,
                "result": result.dict()
            })
        
        return {"batch_results": results}
        
    except Exception as e:
        logger.error(f"Error in batch verification: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Batch verification failed: {str(e)}")

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "GestPay FacePay API is running",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "register_face": "/register-face",
            "verify_face": "/verify-face",
            "batch_verify": "/batch-verify"
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
