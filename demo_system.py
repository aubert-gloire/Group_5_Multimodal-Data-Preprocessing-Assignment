"""
User Identity and Product Recommendation System
Demonstrates the complete authentication and prediction flow:
Start → Face Recognition → Product Recommendation → Voice Validation → Display Product
"""
import os
import sys
import argparse
import joblib
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
import warnings
warnings.filterwarnings('ignore')

# Import deep learning libraries
try:
    from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
    import librosa
except ImportError as e:
    print(f"Missing required library: {e}")
    print("Install with: pip install tensorflow librosa pillow")
    sys.exit(1)

# Paths
ROOT = Path(__file__).parent
MODELS_DIR = ROOT / 'models'
DATASET_DIR = ROOT / 'Dataset'

# Global variables for models 
face_model = None
face_scaler = None
face_encoder = None
mobilenet = None
product_pipeline = None
product_encoder = None
voice_model = None
audio_scaler = None

# Load all models at startup
print("Loading models...")
try:
    # Facial Recognition Model
    face_model = joblib.load(MODELS_DIR / 'face_auth_rf.joblib')
    face_scaler = joblib.load(MODELS_DIR / 'face_auth_scaler.joblib')
    face_encoder = joblib.load(MODELS_DIR / 'face_auth_label_encoder.joblib')
    mobilenet = MobileNetV2(weights='imagenet', include_top=False, pooling='avg', input_shape=(224, 224, 3))
    
    # Product Recommendation Model
    product_pipeline = joblib.load(MODELS_DIR / 'product_recommendation_pipeline.joblib')
    product_encoder = joblib.load(MODELS_DIR / 'product_label_encoder.joblib')
    
    # Voiceprint Model
    with open(MODELS_DIR / 'voiceprint_model.pkl', 'rb') as f:
        voice_model = pickle.load(f)
    
    # Audio Scaler (required for voice model)
    try:
        with open(MODELS_DIR / 'audio_scaler.pkl', 'rb') as f:
            audio_scaler = pickle.load(f)
        print("✓ Audio scaler loaded")
    except FileNotFoundError:
        print("⚠️  Warning: audio_scaler.pkl not found. Voice validation may fail.")
        print("   Run the audio processing notebook to generate the scaler.")
        audio_scaler = None
    
    print("✓ All models loaded successfully\n")
except Exception as e:
    print(f"Error loading models: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)


def extract_face_features(image_path):
    """Extract facial features using MobileNetV2"""
    try:
        img = Image.open(image_path).convert('RGB').resize((224, 224))
        img_array = np.array(img)
        img_array = preprocess_input(img_array.astype(np.float32))
        img_array = np.expand_dims(img_array, axis=0)
        features = mobilenet.predict(img_array, verbose=0)[0]
        return features
    except Exception as e:
        print(f"Error processing image: {e}")
        return None


def facial_recognition(image_path, threshold=0.6):
    """Step 1: Facial Recognition Model"""
    print("=" * 60)
    print("STEP 1: FACIAL RECOGNITION")
    print("=" * 60)
    
    if not os.path.exists(image_path):
        print(f"✗ Image not found: {image_path}")
        return None, False
    
    print(f"Processing image: {os.path.basename(image_path)}")
    features = extract_face_features(image_path)
    
    if features is None:
        return None, False
    
    # Scale and predict
    features_scaled = face_scaler.transform(features.reshape(1, -1))
    prediction = face_model.predict(features_scaled)[0]
    probabilities = face_model.predict_proba(features_scaled)[0]
    confidence = float(probabilities[prediction])
    
    user_name = face_encoder.inverse_transform([prediction])[0]
    
    print(f"Identified User: {user_name}")
    print(f"Confidence: {confidence:.2%}")
    
    if confidence >= threshold:
        print(f"✓ Face Recognition: PASS (Confidence >= {threshold:.0%})")
        return user_name, True
    else:
        print(f"✗ Face Recognition: FAIL (Low confidence)")
        return user_name, False


def run_product_recommendation(user_data):
    """Step 2: Run Product Recommendation Model"""
    print("\n" + "=" * 60)
    print("STEP 2: PRODUCT RECOMMENDATION")
    print("=" * 60)
    
    try:
        # Required features for product recommendation model
        required_features = [
            'social_media_platform',
            'engagement_score',
            'purchase_interest_score',
            'review_sentiment',
            'purchase_amount',
            'customer_rating'
        ]
        
        
        df_user = pd.DataFrame([user_data])
        
        # Ensure all required features are present
        missing_features = [f for f in required_features if f not in df_user.columns]
        if missing_features:
            print(f"⚠️  Warning: Missing features: {missing_features}")
            print("   Using default values for missing features")
            for feat in missing_features:
                if feat == 'social_media_platform':
                    df_user[feat] = 'Twitter'
                elif feat == 'review_sentiment':
                    df_user[feat] = 'Neutral'
                else:
                    df_user[feat] = 0.0
        
        # Select only required features (in correct order)
        df_user = df_user[required_features]
        
        # Make prediction
        prediction = product_pipeline.predict(df_user)[0]
        probabilities = product_pipeline.predict_proba(df_user)[0]
        
        product = product_encoder.inverse_transform([prediction])[0]
        confidence = float(probabilities[prediction])
        
        print(f"Predicted Product: {product}")
        print(f"Confidence: {confidence:.2%}")
        
        return product, confidence, True
    except Exception as e:
        print(f"✗ Product Recommendation Error: {e}")
        import traceback
        traceback.print_exc()
        return None, 0, False


def extract_audio_features(audio_path):
    """Extract audio features matching training (38 features total)"""
    try:
        y, sr = librosa.load(audio_path, sr=None)
        features = {}
        
        # 13 MFCCs (mean + std = 26 features)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        for i in range(13):
            features[f'mfcc_{i+1}_mean'] = np.mean(mfccs[i])
            features[f'mfcc_{i+1}_std'] = np.std(mfccs[i])
        
        # Spectral rolloff (mean + std = 2 features)
        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
        features['spectral_rolloff_mean'] = np.mean(spectral_rolloff)
        features['spectral_rolloff_std'] = np.std(spectral_rolloff)
        
        # RMS energy (mean + std = 2 features)
        rms = librosa.feature.rms(y=y)
        features['rms_energy_mean'] = np.mean(rms)
        features['rms_energy_std'] = np.std(rms)
        
        # Zero crossing rate (mean + std = 2 features)
        zcr = librosa.feature.zero_crossing_rate(y)
        features['zero_crossing_rate_mean'] = np.mean(zcr)
        features['zero_crossing_rate_std'] = np.std(zcr)
        
        # Spectral centroid (mean + std = 2 features)
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
        features['spectral_centroid_mean'] = np.mean(spectral_centroid)
        features['spectral_centroid_std'] = np.std(spectral_centroid)
        
        # Chroma (mean + std = 2 features)
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        features['chroma_mean'] = np.mean(chroma)
        features['chroma_std'] = np.std(chroma)
        
        # Spectral bandwidth (mean + std = 2 features)
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
        features['spectral_bandwidth_mean'] = np.mean(spectral_bandwidth)
        features['spectral_bandwidth_std'] = np.std(spectral_bandwidth)
        
        # Convert to DataFrame with correct column order (must match training)
        feature_cols = [f'mfcc_{i+1}_mean' for i in range(13)] + \
                       [f'mfcc_{i+1}_std' for i in range(13)] + \
                       ['spectral_rolloff_mean', 'spectral_rolloff_std',
                        'rms_energy_mean', 'rms_energy_std',
                        'zero_crossing_rate_mean', 'zero_crossing_rate_std',
                        'spectral_centroid_mean', 'spectral_centroid_std',
                        'chroma_mean', 'chroma_std',
                        'spectral_bandwidth_mean', 'spectral_bandwidth_std']
        
        return pd.DataFrame([features])[feature_cols]
    except Exception as e:
        print(f"Error processing audio: {e}")
        import traceback
        traceback.print_exc()
        return None


def voice_validation(audio_path, expected_user=None):
    """Step 3: Voice Validation Model (binary: authorized=1, unauthorized=0)"""
    print("\n" + "=" * 60)
    print("STEP 3: VOICE VALIDATION")
    print("=" * 60)
    
    if not os.path.exists(audio_path):
        print(f"✗ Audio file not found: {audio_path}")
        return False
    
    if audio_scaler is None:
        print("✗ Audio scaler not loaded. Cannot perform voice validation.")
        return False
    
    print(f"Processing audio: {os.path.basename(audio_path)}")
    features = extract_audio_features(audio_path)
    
    if features is None:
        return False
    
    try:
        # Scale features before prediction 
        features_scaled = audio_scaler.transform(features)
        
        # Voice model is binary: 0=unauthorized, 1=authorized
        prediction = voice_model.predict(features_scaled)[0]
        probabilities = voice_model.predict_proba(features_scaled)[0]
        confidence = float(probabilities[prediction])
        
        is_authorized = bool(prediction)
        
        print(f"Voice Verification Result: {'AUTHORIZED' if is_authorized else 'UNAUTHORIZED'}")
        print(f"Confidence: {confidence:.2%}")
        
        if is_authorized:
            print(f"✓ Voice Validation: APPROVED")
            return True
        else:
            print(f"✗ Voice Validation: REJECTED (Unauthorized speaker)")
            return False
    except Exception as e:
        print(f"✗ Voice Validation Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def display_final_result(product, confidence):
    """Step 4: Display Predicted Product"""
    print("\n" + "=" * 60)
    print("FINAL RESULT")
    print("=" * 60)
    print(f"✓ AUTHENTICATION SUCCESSFUL")
    print(f"\n Recommended Product: {product}")
    print(f" Confidence Score: {confidence:.2%}")
    print("=" * 60)


def access_denied(stage):
    """Display access denied message"""
    print("\n" + "=" * 60)
    print("ACCESS DENIED")
    print("=" * 60)
    print(f"✗ Authentication failed at: {stage}")
    print("User is not authorized to access the system.")
    print("=" * 60)


def simulate_authorized_flow(image_path, audio_path, user_data):
    """Simulate a complete authorized transaction"""
    print("\n Starting User Identity and Product Recommendation System\n")
    
    # Step 1: Facial Recognition
    user_name, face_passed = facial_recognition(image_path)
    if not face_passed:
        access_denied("Facial Recognition")
        return False
    
    # Step 2: Product Recommendation (runs after face passes)
    product, confidence, product_passed = run_product_recommendation(user_data)
    if not product_passed:
        access_denied("Product Recommendation")
        return False
    
    # Step 3: Voice Validation (must approve before showing product)
    voice_passed = voice_validation(audio_path, expected_user=user_name)
    if not voice_passed:
        access_denied("Voice Validation")
        return False
    
    # Step 4: Display Product (only if all checks pass)
    display_final_result(product, confidence)
    return True


def simulate_unauthorized_flow(image_path, audio_path, user_data, fail_at='face'):
    """Simulate an unauthorized attempt"""
    print("\n🔐 Starting User Identity and Product Recommendation System")
    print(f"⚠️  Simulating UNAUTHORIZED attempt (will fail at {fail_at})\n")
    
    if fail_at == 'face':
        print("=" * 60)
        print("STEP 1: FACIAL RECOGNITION")
        print("=" * 60)
        print("✗ Face not recognized or confidence too low")
        access_denied("Facial Recognition")
        return False
    
    # If not failing at face, proceed normally
    user_name, face_passed = facial_recognition(image_path)
    if not face_passed:
        access_denied("Facial Recognition")
        return False
    
    product, confidence, product_passed = run_product_recommendation(user_data)
    if not product_passed:
        access_denied("Product Recommendation")
        return False
    
    if fail_at == 'voice':
        print("\n" + "=" * 60)
        print("STEP 3: VOICE VALIDATION")
        print("=" * 60)
        print("✗ Voice does not match or unauthorized speaker")
        access_denied("Voice Validation")
        return False
    
    return True


def main():
    parser = argparse.ArgumentParser(description='User Identity and Product Recommendation System Demo')
    parser.add_argument('--image', type=str, help='Path to facial image')
    parser.add_argument('--audio', type=str, help='Path to voice sample')
    parser.add_argument('--user-data', type=str, help='Path to CSV file with user data for product prediction')
    parser.add_argument('--row-index', type=int, default=0, help='Row index to use from user-data CSV')
    parser.add_argument('--unauthorized', action='store_true', help='Simulate unauthorized attempt')
    parser.add_argument('--fail-at', choices=['face', 'voice'], default='face', help='Where to fail in unauthorized mode')
    
    args = parser.parse_args()
    
    # Load user data for product recommendation
    if args.user_data:
        df = pd.read_csv(args.user_data)
        user_data = df.iloc[args.row_index].to_dict()
    else:
        # Load a real sample from the merged dataset
        try:
            df = pd.read_csv(DATASET_DIR / 'merged_customer_data.csv')
            user_data = df.iloc[0].to_dict()
            # Remove the target column if present
            if 'product_category' in user_data:
                del user_data['product_category']
            print(f"Loaded user data from merged_customer_data.csv (row 0)")
        except Exception as e:
            print(f"Warning: Could not load default user data: {e}")
            # Fallback minimal data
            user_data = {
                'social_media_platform': 'Instagram',
                'engagement_score': 75.0,
                'purchase_interest_score': 80.0,
                'review_sentiment': 'Positive',
                'purchase_amount': 150.0,
                'customer_rating': 4.5
            }
    
    # Set default paths if not provided
    if not args.image:
        args.image = str(ROOT / 'Notebook' / 'group_images' / 'Aubert_neutral.jpeg')
    if not args.audio:
        args.audio = str(DATASET_DIR / 'audio_samples' / 'original' / 'Aubert confirm transaction.wav')
    
    # Run the flow
    if args.unauthorized:
        simulate_unauthorized_flow(args.image, args.audio, user_data, fail_at=args.fail_at)
    else:
        simulate_authorized_flow(args.image, args.audio, user_data)


if __name__ == '__main__':
    main()
