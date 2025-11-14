"""
Simple CLI to predict product category using the trained pipeline.
Usage examples:
  python scripts/predict_product.py --interactive
  python scripts/predict_product.py --csv-row path/to/file.csv --row-index 0
"""
import os
import argparse
import pandas as pd

# Try to load joblib artifacts; fall back to a lightweight dummy model module
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
MODELS_DIR = os.path.join(ROOT, 'models')
PIPELINE_PATH = os.path.join(MODELS_DIR, 'product_recommendation_pipeline.joblib')
ENCODER_PATH = os.path.join(MODELS_DIR, 'product_label_encoder.joblib')

pipe = None
le = None
try:
    import joblib
    if os.path.exists(PIPELINE_PATH) and os.path.exists(ENCODER_PATH):
        pipe = joblib.load(PIPELINE_PATH)
        le = joblib.load(ENCODER_PATH)
    else:
        raise FileNotFoundError('joblib artifacts missing')
except Exception:
    # fallback: import the dummy model module placed in models/
    try:
        from models import dummy_product_model as dm
        pipe = dm.pipeline
        le = dm.label_encoder
        print('Warning: using dummy model from models/dummy_product_model.py')
    except Exception:
        raise RuntimeError('No model artifacts available (joblib files missing and dummy model import failed)')

FEATURES = [
    'social_media_platform',
    'engagement_score',
    'purchase_interest_score',
    'review_sentiment',
    'purchase_amount',
    'customer_rating'
]


def predict_from_series(s: pd.Series):
    x = s[FEATURES].to_frame().T
    pred_idx = pipe.predict(x)[0]
    pred_label = le.inverse_transform([pred_idx])[0]
    return pred_label


def simulate_face_check(allow=True):
    """Simulate a face recognition check. Return True if allowed, False otherwise."""
    return allow


def simulate_voice_check(allow=True):
    """Simulate a voice approval check. Return True if allowed, False otherwise."""
    return allow


def interactive_prompt():
    data = {}
    data['social_media_platform'] = input('social_media_platform (e.g., Facebook, Twitter, Instagram, TikTok, LinkedIn): ').strip()
    data['engagement_score'] = float(input('engagement_score (numeric): ').strip())
    data['purchase_interest_score'] = float(input('purchase_interest_score (numeric): ').strip())
    data['review_sentiment'] = input('review_sentiment (Positive/Neutral/Negative): ').strip()
    data['purchase_amount'] = float(input('purchase_amount (numeric): ').strip())
    data['customer_rating'] = float(input('customer_rating (numeric 0-5): ').strip())
    s = pd.Series(data)
    pred = predict_from_series(s)
    print('Predicted product category:', pred)


def predict_from_csv_row(csv_path, row_index=0):
    df = pd.read_csv(csv_path)
    if row_index < 0 or row_index >= len(df):
        raise IndexError('row_index out of range')
    s = df.iloc[row_index]
    pred = predict_from_series(s)
    print('Predicted product category for row', row_index, ':', pred)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--interactive', action='store_true', help='Prompt for feature values interactively')
    parser.add_argument('--csv-row', type=str, help='Path to CSV and predict from a row')
    parser.add_argument('--row-index', type=int, default=0, help='Row index for --csv-row')
    parser.add_argument('--require-face', action='store_true', help='Require simulated face auth before predicting')
    parser.add_argument('--require-voice', action='store_true', help='Require simulated voice approval before predicting')
    parser.add_argument('--force-unauthorized', action='store_true', help='Simulate an unauthorized attempt (deny auth)')
    args = parser.parse_args()

    if args.interactive:
        # interactive flow: optionally require auth
        if args.require_face or args.require_voice:
            allowed = True
            if args.require_face:
                allowed = simulate_face_check(not args.force_unauthorized)
            if allowed and args.require_voice:
                allowed = simulate_voice_check(not args.force_unauthorized)
            if not allowed:
                print('Access denied: authentication failed')
                exit(1)
        interactive_prompt()
    elif args.csv_row:
        # when predicting from CSV row, optionally enforce simulated auth
        if args.require_face or args.require_voice:
            allowed = True
            if args.require_face:
                allowed = simulate_face_check(not args.force_unauthorized)
            if allowed and args.require_voice:
                allowed = simulate_voice_check(not args.force_unauthorized)
            if not allowed:
                print('Access denied: authentication failed')
                exit(1)
        predict_from_csv_row(args.csv_row, args.row_index)
    else:
        print('No mode selected. Use --interactive or --csv-row')
