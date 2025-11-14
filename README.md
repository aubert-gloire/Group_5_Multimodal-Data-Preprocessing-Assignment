Product Recommendation Model - quick guide

Files added:
- `scripts/train_product_model.py` : trains a RandomForest pipeline on `Dataset/merged_customer_data.csv` and saves artifacts to `models/`
- `scripts/predict_product.py` : simple CLI to run predictions using the trained pipeline
- `scripts/cli_full_flow.py` : combined face/voice auth simulation then prediction
- `models/dummy_product_model.py` : fallback dummy pipeline for demos
- `requirements.txt` : Python dependencies

Quick steps:
1) Create a Python environment and install deps:
   `python -m venv .venv; .\\.venv\\Scripts\\Activate.ps1; pip install -r requirements.txt`

2) Train the model:
   `python scripts/train_product_model.py`

   This will produce:
   - `models/product_recommendation_pipeline.joblib`
   - `models/product_label_encoder.joblib`

3) Run an interactive prediction:
   `python scripts/predict_product.py --interactive`
   # Simulate auth checks (face + voice):
   `python scripts/predict_product.py --interactive --require-face --require-voice`

4) Or predict for a specific CSV row (e.g., the merged dataset):
   `python scripts/predict_product.py --csv-row Dataset/merged_customer_data.csv --row-index 0`
   # With auth required (simulate allow):
   `python scripts/predict_product.py --csv-row Dataset/merged_customer_data.csv --row-index 0 --require-face --require-voice`
   # Simulate unauthorized attempt (will deny):
   `python scripts/predict_product.py --csv-row Dataset/merged_customer_data.csv --row-index 0 --require-face --require-voice --force-unauthorized`

5) Full simulated flow (face -> voice -> predict) via combined CLI:
   `python scripts/cli_full_flow.py --csv-row Dataset/merged_customer_data.csv --row-index 0 --face-allow --voice-allow`
   `python scripts/cli_full_flow.py --interactive --face-deny`  # will deny at face check

Notes and next steps:
- The training script uses a minimal selected feature set. Extend feature engineering (temporal features from `purchase_date`, user-level aggregations, image/audio features) for better accuracy.
- For reproducibility in your assignment, consider saving evaluation outputs (confusion matrix, classification report) to files.
- Real face and voice authentication is simulated in the CLI. To implement real biometric checks you will need labeled image/audio data, trained models, and privacy safeguards.
Product Recommendation Model - quick guide\n\nFiles added:\n- scripts/train_product_model.py : trains a RandomForest pipeline on Dataset/merged_customer_data.csv and saves artifacts to models/\n- scripts/predict_product.py : simple CLI to run predictions using the trained pipeline\n- requirements.txt : Python dependencies\n\nQuick steps:\n1) Create a Python environment and install deps:\n   python -m venv .venv; .\.venv\Scripts\Activate.ps1; pip install -r requirements.txt\n\n2) Train the model:\n   python scripts/train_product_model.py\n\n   This will produce:\n   - models/product_recommendation_pipeline.joblib\n   - models/product_label_encoder.joblib\n\n3) Run an interactive prediction:\n   python scripts/predict_product.py --interactive\n\n4) Or predict for a specific CSV row (e.g., the merged dataset):\n   python scripts/predict_product.py --csv-row Dataset/merged_customer_data.csv --row-index 0\n\nNotes and next steps:\n- The training script uses a minimal selected feature set. You can extend feature engineering (temporal features from purchase_date, user-level aggregations, image/audio features) for better accuracy.\n- For reproducibility in your assignment, consider saving evaluation outputs (confusion matrix, classification report) to files.\n- To simulate unauthorized attempts, call the prediction CLI with manipulated inputs to reflect unknown/incorrect user attributes or provide a user id that doesn't match saved profiles (that part of the pipeline will be part of your face/voice auth components).\n