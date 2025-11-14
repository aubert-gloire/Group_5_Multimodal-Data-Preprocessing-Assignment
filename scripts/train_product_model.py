"""
Train a product recommendation (classification) model on Dataset/merged_customer_data.csv
Saves artifacts to models/product_recommendation_pipeline.joblib and models/label_encoder.joblib
"""
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report
import joblib

# Paths
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATA_PATH = os.path.join(ROOT, 'Dataset', 'merged_customer_data.csv')
MODELS_DIR = os.path.join(ROOT, 'models')
os.makedirs(MODELS_DIR, exist_ok=True)

def load_data(path=DATA_PATH):
    df = pd.read_csv(path)

    # Try to merge precomputed audio/image features if they exist
    audio_path = os.path.join(os.path.dirname(path), 'audio_features.csv')
    image_path = os.path.join(os.path.dirname(path), 'image_features.csv')

    if os.path.exists(audio_path):
        try:
            audio_df = pd.read_csv(audio_path)
            # Expect audio_features to have a customer id column; try common names
            key_cols = ['customer_id_new', 'customer_id', 'customer_id_legacy']
            join_col = next((c for c in key_cols if c in audio_df.columns and c in df.columns), None)
            if join_col:
                df = df.merge(audio_df, on=join_col, how='left')
                print(f'Merged audio features on {join_col}, new shape: {df.shape}')
        except Exception as e:
            print('Warning: failed to merge audio features:', e)

    if os.path.exists(image_path):
        try:
            img_df = pd.read_csv(image_path)
            key_cols = ['customer_id_new', 'customer_id', 'customer_id_legacy']
            join_col = next((c for c in key_cols if c in img_df.columns and c in df.columns), None)
            if join_col:
                df = df.merge(img_df, on=join_col, how='left')
                print(f'Merged image features on {join_col}, new shape: {df.shape}')
        except Exception as e:
            print('Warning: failed to merge image features:', e)

    return df


def preprocess_and_train(df):
    # Target: product_category
    if 'product_category' not in df.columns:
        raise ValueError('Expected column product_category in dataset')

    # Minimal feature set selected from dataset
    features = [
        'social_media_platform',
        'engagement_score',
        'purchase_interest_score',
        'review_sentiment',
        'purchase_amount',
        'customer_rating'
    ]
    df = df.copy()

    # keep only features we want and drop rows with missing target
    df = df.dropna(subset=['product_category'])

    X = df[features]
    y = df['product_category']

    # Simple cleaning: coerce numeric columns
    numeric_features = ['engagement_score', 'purchase_interest_score', 'purchase_amount', 'customer_rating']
    categorical_features = ['social_media_platform', 'review_sentiment']

    # Column transformer
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features),
    ])

    # Model pipeline
    clf = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1))
    ])

    # Encode target
    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y_enc, test_size=0.2, random_state=42, stratify=y_enc)

    print('Training model...')
    clf.fit(X_train, y_train)

    print('Evaluating...')
    preds = clf.predict(X_test)
    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds, average='macro')
    report = classification_report(y_test, preds, target_names=le.classes_)

    print(f'Accuracy: {acc:.4f}')
    print(f'F1 (macro): {f1:.4f}')
    print('Classification report:\n', report)

    # Save artifacts
    pipeline_path = os.path.join(MODELS_DIR, 'product_recommendation_pipeline.joblib')
    encoder_path = os.path.join(MODELS_DIR, 'product_label_encoder.joblib')
    joblib.dump(clf, pipeline_path)
    joblib.dump(le, encoder_path)
    print(f'Saved pipeline to {pipeline_path}')
    print(f'Saved label encoder to {encoder_path}')

    # Return artifacts for optional programmatic use
    return clf, le


if __name__ == '__main__':
    df = load_data()
    preprocess_and_train(df)
