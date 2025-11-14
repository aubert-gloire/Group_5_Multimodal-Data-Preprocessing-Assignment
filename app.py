"""
FastAPI inference server for face-authentication pipeline.

- Loads MobileNetV2 (for embeddings) and your saved artifacts from `models/` (joblib files).
- Attempts to load `models/face_pipeline_with_le.joblib` (dict with 'pipeline' and 'label_encoder').
  If not found, falls back to loading scaler + classifier + label encoder separately.

Endpoints:
- GET /health -> {"status":"ok"}
- POST /predict -> multipart/form-data file=<image>

Usage (dev):
    & .\venv\Scripts\Activate.ps1
    pip install -r requirements.txt
    uvicorn app:app --host 0.0.0.0 --port 8000

Notes:
- This service expects joblib artifacts (not .pkl). It will NOT convert or rename your files.
- Keep large model files out of git; place them in `models/` or download via provided script.
"""
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn
import os
import io
from PIL import Image
import numpy as np
import joblib
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input

from scipy.spatial.distance import euclidean

app = FastAPI(title="Face Authentication Service")

# Load embedding model (MobileNetV2)
mobilenet_model = None
try:
    mobilenet_model = MobileNetV2(weights='imagenet', include_top=False, pooling='avg', input_shape=(224,224,3))
except Exception as e:
    # don't crash the entire import; delay until first request if necessary
    mobilenet_model = None
    print('Warning: MobileNet model not loaded at startup:', e)

# Try to load a combined artifact first
PIPELINE_CANDIDATES = [
    os.path.join('models', 'face_pipeline_with_le.joblib'),
    os.path.join('models', 'face_pipeline.joblib')
]

ARTIFACT = {'pipeline': None, 'label_encoder': None, 'scaler': None, 'clf': None}

for p in PIPELINE_CANDIDATES:
    if os.path.exists(p):
        loaded = joblib.load(p)
        # If the artifact is a dict containing pipeline + label_encoder
        if isinstance(loaded, dict) and 'pipeline' in loaded:
            ARTIFACT['pipeline'] = loaded['pipeline']
            ARTIFACT['label_encoder'] = loaded.get('label_encoder')
        else:
            # loaded may be a Pipeline
            ARTIFACT['pipeline'] = loaded
        break

# If no pipeline, try separate files
if ARTIFACT['pipeline'] is None:
    clf_path = None
    for cand in ['face_recognition_model.joblib', 'face_auth_logreg.joblib', 'face_auth_rf.joblib', 'face_auth_xgboost.joblib']:
        p = os.path.join('models', cand)
        if os.path.exists(p):
            clf_path = p
            break
    if clf_path:
        ARTIFACT['clf'] = joblib.load(clf_path)
    scaler_path = os.path.join('models', 'face_auth_scaler.joblib')
    if os.path.exists(scaler_path):
        ARTIFACT['scaler'] = joblib.load(scaler_path)
    le_path = os.path.join('models', 'face_auth_label_encoder.joblib')
    if os.path.exists(le_path):
        ARTIFACT['label_encoder'] = joblib.load(le_path)


def ensure_mobilenet():
    global mobilenet_model
    if mobilenet_model is None:
        mobilenet_model = MobileNetV2(weights='imagenet', include_top=False, pooling='avg', input_shape=(224,224,3))


def load_image_bytes(data: bytes) -> np.ndarray:
    image = Image.open(io.BytesIO(data)).convert('RGB')
    image = image.resize((224,224))
    arr = np.array(image).astype(np.float32)
    arr = preprocess_input(arr)
    arr = np.expand_dims(arr, axis=0)
    return arr


@app.get('/health')
def health():
    return {"status": "ok"}


class PredictResponse(BaseModel):
    predicted: str
    confidence: float
    probabilities: dict
    distance: float | None = None
    closest: str | None = None


@app.post('/predict', response_model=PredictResponse)
async def predict(file: UploadFile = File(...)):
    # Read image
    try:
        contents = await file.read()
        img_tensor = load_image_bytes(contents)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f'Invalid image: {e}')

    # Ensure mobilenet loaded
    try:
        ensure_mobilenet()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f'Failed to load embedding model: {e}')

    # Extract embedding
    try:
        emb = mobilenet_model.predict(img_tensor)[0]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f'Error computing embedding: {e}')

    # Predict using pipeline or separate scaler+clf
    if ARTIFACT['pipeline'] is not None:
        probs = ARTIFACT['pipeline'].predict_proba(np.expand_dims(emb, axis=0))[0]
        classes = ARTIFACT['pipeline'].classes_ if hasattr(ARTIFACT['pipeline'], 'classes_') else ARTIFACT['pipeline'].named_steps['clf'].classes_
    else:
        if ARTIFACT['scaler'] is not None:
            emb_in = ARTIFACT['scaler'].transform([emb])
        else:
            emb_in = np.expand_dims(emb, axis=0)
        if ARTIFACT['clf'] is None:
            raise HTTPException(status_code=500, detail='No classifier found on server')
        probs = ARTIFACT['clf'].predict_proba(emb_in)[0]
        classes = ARTIFACT['clf'].classes_

    # Build probabilities dict
    prob_dict = {str(c): float(p) for c, p in zip(classes, probs)}
    top_idx = int(np.argmax(probs))
    predicted = classes[top_idx]
    confidence = float(probs[top_idx])

    # Decode label if label encoder present
    decoded_pred = predicted
    if ARTIFACT.get('label_encoder') is not None:
        try:
            le = ARTIFACT['label_encoder']
            # classifier classes_ may be numeric indices or label strings
            if isinstance(predicted, (int, np.integer)):
                decoded_pred = le.inverse_transform([predicted])[0]
            else:
                # predicted is already a label
                decoded_pred = str(predicted)
        except Exception:
            decoded_pred = str(predicted)

    # Optional distance check against known features
    distance = None
    closest_name = None
    known_features_path = os.path.join('models', 'known_features.npz')
    if os.path.exists(known_features_path):
        try:
            data = np.load(known_features_path)
            known_X = data['X_train']
            known_y = data['y_train']
            dists = [float(euclidean(emb, k)) for k in known_X]
            min_idx = int(np.argmin(dists))
            distance = float(dists[min_idx])
            closest_label = known_y[min_idx]
            if ARTIFACT.get('label_encoder') is not None:
                try:
                    closest_name = ARTIFACT['label_encoder'].inverse_transform([int(closest_label)])[0]
                except Exception:
                    closest_name = str(closest_label)
            else:
                closest_name = str(closest_label)
        except Exception:
            distance = None
            closest_name = None

    return JSONResponse(
        content={
            'predicted': str(decoded_pred),
            'confidence': confidence,
            'probabilities': prob_dict,
            'distance': distance,
            'closest': closest_name,
        }
    )


if __name__ == '__main__':
    uvicorn.run('app:app', host='0.0.0.0', port=8000, reload=False)
