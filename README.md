# Multimodal User Authentication and Product Recommendation System

This project implements a complete authentication and product recommendation pipeline using multimodal data: facial images, voice recordings, and customer transaction data. The system follows a sequential authentication flow where users must pass facial recognition and voice validation before receiving personalized product recommendations.

## Project Structure

```
Group_5_Multimodal-Data-Preprocessing-Assignment/
├── Dataset/
│   ├── customer_social_profiles - customer_social_profiles.csv
│   ├── customer_transactions - customer_transactions.csv
│   ├── merged_customer_data.csv          # Output from merge_datasets.ipynb
│   ├── audio_features.csv                # Output from audio processing notebook
│   ├── image_features_clean.csv          # Output from facial feature extraction
│   └── audio_samples/
│       ├── original/                     # Original voice recordings
│       ├── augmented/                    # Augmented audio samples
│       └── unauthorized/                 # Unauthorized voice samples for testing
├── models/
│   ├── face_auth_logreg.joblib          # Logistic Regression facial model
│   ├── face_auth_rf.joblib              # Random Forest facial model
│   ├── face_auth_xgboost.joblib         # XGBoost facial model
│   ├── face_auth_scaler.joblib          # Feature scaler for facial recognition
│   ├── face_auth_label_encoder.joblib   # Label encoder for user names
│   ├── known_features.npz               # Known facial features database
│   ├── voiceprint_model.pkl             # Logistic Regression voiceprint model
│   ├── product_recommendation_pipeline.joblib  # Product recommendation pipeline
│   └── product_label_encoder.joblib     # Product category encoder
├── Notebook/
│   ├── facial_nb.ipynb                  # Facial feature extraction using MobileNetV2
│   ├── face-authentication_model.ipynb  # Train facial recognition models
│   ├── audio_processing_and_voiceprint_model.ipynb  # Audio processing and voiceprint training
│   ├── merge_datasets.ipynb             # Merge customer social and transaction data
│   ├── train_product_recommender.ipynb  # Train product recommendation model
│   └── group_images/                    # Team member facial images
│       ├── Aubert_neutral.jpeg
│       ├── Aubert_smiling.jpeg
│       ├── Aubert_surprised.jpeg
│       └── [other member images]
├── demo_system.py                        # Command-line demonstration application
├── requirements.txt                      # Python dependencies
└── README.md
```

## System Requirements

- Python 3.8 or higher
- Windows PowerShell (paths use Windows conventions)
- 4GB RAM minimum for model training
- 2GB disk space for datasets and models

## Installation and Setup

### Step 1: Clone Repository

```powershell
git clone https://github.com/aubert-gloire/Group_5_Multimodal-Data-Preprocessing-Assignment.git
cd Group_5_Multimodal-Data-Preprocessing-Assignment
```

### Step 2: Create Virtual Environment

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### Step 3: Install Dependencies

```powershell
pip install tensorflow pandas numpy scikit-learn xgboost librosa pillow joblib matplotlib seaborn jupyter
```

Required packages:
- tensorflow: MobileNetV2 for facial feature extraction
- pandas: Data manipulation and CSV handling
- numpy: Numerical computations
- scikit-learn: Machine learning models and preprocessing
- xgboost: Gradient boosting classifier
- librosa: Audio feature extraction
- pillow: Image processing
- joblib: Model serialization
- matplotlib, seaborn: Data visualization
- jupyter: Notebook execution

## Training Pipeline

Execute the following notebooks in order to train all models:

### 1. Data Merging

Open and run `Notebook/merge_datasets.ipynb`

This notebook:
- Loads customer social profiles and transaction data
- Removes duplicates from both datasets
- Standardizes customer_id fields for proper merging
- Performs inner join on customer_id
- Outputs `Dataset/merged_customer_data.csv`

Key operations:
- Converts customer_id_new (format: A1234) to integer customer_id
- Converts purchase_date to datetime format
- Handles missing values and data type inconsistencies

### 2. Facial Feature Extraction

Open and run `Notebook/facial_nb.ipynb`

This notebook:
- Loads facial images from `Notebook/group_images/`
- Uses MobileNetV2 pretrained on ImageNet to extract 1280-dimensional feature embeddings
- Processes augmented images from `Notebook/augmented_images/`
- Outputs `Dataset/image_features_clean.csv` and `models/image_features_clean.npz`

Image requirements:
- Each team member must have at least 3 images: neutral, smiling, surprised expressions
- Images should be in JPEG format
- Recommended resolution: 224x224 pixels or larger

### 3. Facial Authentication Model Training

Open and run `Notebook/face-authentication_model.ipynb`

This notebook:
- Loads image features from `Dataset/image_features_clean.csv`
- Trains three classification models:
  - Logistic Regression
  - Random Forest (200 estimators)
  - XGBoost
- Evaluates models using accuracy, precision, recall, F1-score
- Saves trained models to `models/` directory

Outputs:
- `models/face_auth_logreg.joblib`
- `models/face_auth_rf.joblib`
- `models/face_auth_xgboost.joblib`
- `models/face_auth_scaler.joblib`
- `models/face_auth_label_encoder.joblib`
- `models/known_features.npz`

### 4. Audio Processing and Voiceprint Model

Open and run `Notebook/audio_processing_and_voiceprint_model.ipynb`

This notebook:
- Loads audio samples from `Dataset/audio_samples/original/`
- Applies augmentations: pitch shift, time stretch, background noise
- Extracts 38 audio features per sample:
  - 13 MFCCs (mean and standard deviation)
  - Spectral rolloff (mean and std)
  - RMS energy (mean and std)
  - Zero crossing rate (mean and std)
  - Spectral centroid (mean and std)
  - Chroma features (mean and std)
  - Spectral bandwidth (mean and std)
- Trains Logistic Regression classifier for voice authentication
- Outputs `Dataset/audio_features.csv` and `models/voiceprint_model.pkl`

Audio requirements:
- Each team member records 2 phrases: "Yes, approve" and "Confirm transaction"
- WAV format recommended
- Sample rate: 16kHz or higher
- Duration: 2-5 seconds per sample

### 5. Product Recommendation Model

Open and run `Notebook/train_product_recommender.ipynb`

This notebook:
- Loads `Dataset/merged_customer_data.csv`
- Engineers temporal features from purchase_date: purchase_month, purchase_day
- Calculates aggregated features: mean_purchase_amount, transaction_count
- Builds preprocessing pipeline with StandardScaler and OneHotEncoder
- Trains Random Forest classifier (200 estimators) to predict product_category
- Evaluates using accuracy, F1-score, log loss, classification report
- Outputs models to `models/` directory

Outputs:
- `models/product_recommendation_pipeline.joblib`
- `models/product_label_encoder.joblib`

Feature engineering:
- Numeric features: purchase_amount, engagement_score, customer_rating, etc.
- Categorical features: social_media_platform, review_sentiment
- Temporal features: purchase_month, purchase_day
- Aggregated features: mean_purchase_amount, transaction_count

## Running the Demonstration System

The `demo_system.py` script implements the complete authentication flow as specified in the assignment requirements:

```
START → Face Recognition → Product Recommendation → Voice Validation → Display Product
```

### Default Authorized User Flow

```powershell
python demo_system.py
```

This command runs the complete flow using default parameters:
- Image: `Notebook/group_images/Aubert_neutral.jpeg`
- Audio: `Dataset/audio_samples/original/Aubert confirm transaction.wav`
- User data: First row from `Dataset/merged_customer_data.csv`

Expected output:
```
Loading models...
✓ All models loaded successfully

Starting User Identity and Product Recommendation System

============================================================
STEP 1: FACIAL RECOGNITION
============================================================
Processing image: Aubert_neutral.jpeg
Identified User: Aubert
Confidence: 85.50%
✓ Face Recognition: PASS (Confidence >= 60%)

============================================================
STEP 2: PRODUCT RECOMMENDATION
============================================================
Predicted Product: Electronics
Confidence: 72.34%

============================================================
STEP 3: VOICE VALIDATION
============================================================
Processing audio: Aubert confirm transaction.wav
Voice Match Confidence: 88.20%
✓ Voice Validation: APPROVED

============================================================
FINAL RESULT
============================================================
✓ AUTHENTICATION SUCCESSFUL

 Recommended Product: Electronics
 Confidence Score: 72.34%
============================================================
```

### Custom User Testing

Specify custom image and audio files:

```powershell
python demo_system.py --image "Notebook\group_images\Liliane_neutral.jpeg" --audio "Dataset\audio_samples\original\liliane confirm transaction.wav"
```

### Unauthorized Face Test

Simulate facial recognition failure:

```powershell
python demo_system.py --image "path\to\unknown_person.jpg" --unauthorized --fail-at face
```

Expected output shows access denied at facial recognition stage.

### Unauthorized Voice Test

Simulate voice validation failure:

```powershell
python demo_system.py --audio "Dataset\audio_samples\unauthorized\stranger_voice.wav" --unauthorized --fail-at voice
```

Expected output shows access denied at voice validation stage.

### Using Specific CSV Row

Load user data from a specific row in the merged dataset:

```powershell
python demo_system.py --user-data "Dataset\merged_customer_data.csv" --row-index 5
```

## Model Performance Metrics

Performance metrics are displayed during training in each respective notebook.

### Facial Recognition Models
Trained on image features extracted using MobileNetV2. Evaluated using train-test split with stratification.

### Voiceprint Verification Model
Logistic Regression classifier trained on 38 audio features. Distinguishes authorized team members from unauthorized speakers.

### Product Recommendation Model
Random Forest classifier predicting product categories from merged customer social and transaction data. Uses automated feature engineering and sklearn pipelines.

## Authentication Flow Logic

The system implements strict sequential authentication:

1. Facial Recognition runs first
   - Extracts 1280-dimensional features using MobileNetV2
   - Predicts user identity using Random Forest classifier
   - Requires confidence >= 60% to pass
   - If FAIL: Access denied, flow terminates

2. Product Recommendation runs only if face passes
   - Loads user transaction and social data
   - Engineers temporal and aggregated features
   - Predicts product category using trained pipeline
   - If FAIL: Access denied, flow terminates

3. Voice Validation runs only if product recommendation succeeds
   - Extracts 38 audio features from voice sample
   - Validates voice matches expected user
   - Requires confidence >= 50% to pass
   - If FAIL: Access denied, flow terminates

4. Display Product runs only if all three stages pass
   - Shows recommended product category
   - Displays confidence score
   - Marks transaction as authorized

## Troubleshooting

### Model Loading Errors

If models fail to load, ensure all training notebooks have been executed in order:
1. merge_datasets.ipynb
2. facial_nb.ipynb
3. face-authentication_model.ipynb
4. audio_processing_and_voiceprint_model.ipynb
5. train_product_recommender.ipynb

Verify all model files exist in the `models/` directory.

### Audio File Not Found

Check that audio files exist in `Dataset/audio_samples/original/` and match the exact filenames including spaces:
- `Aubert confirm transaction.wav`
- `Aubert yes approve.wav`
- etc.

### Feature Mismatch Errors

If encountering "expecting N features" errors, ensure the demo_system.py feature extraction matches the training notebooks exactly. The voiceprint model expects 38 features and the facial model expects 1280 features.

### Image Processing Errors

Ensure PIL (Pillow) is installed and images are in valid JPEG format. Images must be readable and contain facial data.

## Data Requirements Summary

### Tabular Data
- customer_social_profiles.csv: Social media engagement metrics
- customer_transactions.csv: Purchase history and ratings
- Merged into merged_customer_data.csv with customer_id as key

### Image Data
- Minimum 3 images per team member (neutral, smiling, surprised)
- JPEG format
- Organized in Notebook/group_images/
- Augmented versions in Notebook/augmented_images/

### Audio Data
- Minimum 2 recordings per team member
- Phrases: "Yes, approve" and "Confirm transaction"
- WAV format preferred
- Stored in Dataset/audio_samples/original/
- Augmented versions in Dataset/audio_samples/augmented/
- Unauthorized samples in Dataset/audio_samples/unauthorized/

## Assignment Deliverables Checklist

This repository contains:
- [x] Merged dataset with feature engineering (merged_customer_data.csv)
- [x] Image features CSV (image_features_clean.csv)
- [x] Audio features CSV (audio_features.csv)
- [x] Jupyter notebooks for all training pipelines
- [x] Python script for command-line demonstration (demo_system.py)
- [x] All three trained models (facial, voice, product)
- [x] Model evaluation metrics in notebooks
- [x] Simulation of authorized user flow
- [x] Simulation capability for unauthorized attempts

## Team Members

Project developed by Group 5 for Multimodal Data Preprocessing Assignment.

Team members with facial and voice samples:
- Aubert
- Liliane
- Pauline
- Jade

## License

This project is submitted as part of an academic assignment.