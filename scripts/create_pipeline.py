"""
Create a single sklearn Pipeline artifact for inference.
Usage:
    python scripts/create_pipeline.py --model-dir Notebook --out-dir models

This script will look for scaler + classifier joblib files in --model-dir and
produce `models/face_pipeline.joblib` (Pipeline of scaler + classifier) and
`models/face_pipeline_with_le.joblib` (dict with pipeline + label_encoder when available).

Do NOT commit the produced joblib files to git; instead push them to Releases or S3.
"""
import os
import argparse
import joblib
from sklearn.pipeline import Pipeline

CANDIDATE_MODELS = [
    'face_recognition_model.joblib',
    'face_auth_logreg.joblib',
    'face_auth_rf.joblib',
    'face_auth_xgboost.joblib',
]

CANDIDATE_SCALERS = [
    'face_auth_scaler.joblib',
]

CANDIDATE_LE = [
    'face_auth_label_encoder.joblib',
]


def find_first_existing(dirpath, candidates):
    for name in candidates:
        path = os.path.join(dirpath, name)
        if os.path.exists(path):
            return path
    return None


def main(model_dir, out_dir, pipeline_name='face_pipeline.joblib'):
    os.makedirs(out_dir, exist_ok=True)

    model_path = find_first_existing(model_dir, CANDIDATE_MODELS)
    scaler_path = find_first_existing(model_dir, CANDIDATE_SCALERS)
    le_path = find_first_existing(model_dir, CANDIDATE_LE)

    if model_path is None:
        raise FileNotFoundError(f"No classifier joblib found in {model_dir}. Looked for: {CANDIDATE_MODELS}")

    print('Found model:', model_path)
    model = joblib.load(model_path)

    scaler = None
    if scaler_path is not None:
        print('Found scaler:', scaler_path)
        scaler = joblib.load(scaler_path)
    else:
        print('No scaler found; pipeline will contain classifier only')

    # Build pipeline
    steps = []
    if scaler is not None:
        steps.append(('scaler', scaler))
    steps.append(('clf', model))
    pipeline = Pipeline(steps)

    pipeline_path = os.path.join(out_dir, pipeline_name)
    joblib.dump(pipeline, pipeline_path)
    print('Saved pipeline to:', pipeline_path)

    # Save pipeline + label encoder (if available) in one artifact for convenience
    if le_path is not None:
        try:
            le = joblib.load(le_path)
            combined = {'pipeline': pipeline, 'label_encoder': le}
            combined_path = os.path.join(out_dir, 'face_pipeline_with_le.joblib')
            joblib.dump(combined, combined_path)
            print('Saved pipeline+label-encoder to:', combined_path)
        except Exception as e:
            print('Failed to load or save label encoder:', e)

    print('\nDone. Please do not commit the generated joblib files directly to git.\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-dir', default='Notebook', help='Directory where individual joblib artifacts live')
    parser.add_argument('--out-dir', default='models', help='Output directory for the pipeline artifact')
    parser.add_argument('--pipeline-name', default='face_pipeline.joblib', help='Filename for the pipeline')
    args = parser.parse_args()
    main(args.model_dir, args.out_dir, args.pipeline_name)
