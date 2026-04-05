import os
import joblib
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Paths
ARTIFACTS_DIR = r'd:\MLFINAL\MachineLearningFinalAssignment\TeamAsyncMLProject\artifacts'
MODEL_PATH = os.path.join(ARTIFACTS_DIR, 'best_segment_classifier.pkl')
PREPROCESSOR_PATH = os.path.join(ARTIFACTS_DIR, 'preprocessor.pkl')
FINAL_MODEL_PATH = os.path.join(ARTIFACTS_DIR, 'model.joblib')

# 1. Load artifacts
classifier = joblib.load(MODEL_PATH)
orig_p = joblib.load(PREPROCESSOR_PATH)

# 2. Create index-based preprocessor
# We map the already fitted transformers from the original preprocessor
fixed_p = ColumnTransformer(
    transformers=[
        ('num', orig_p.named_transformers_['num'], [0, 1, 2, 3, 4]),
        ('cat', orig_p.named_transformers_['cat'], [5, 6, 7, 8])
    ])

# --- HACK TO MARK AS FITTED ---
fixed_p.transformers_ = [
    ('num', orig_p.named_transformers_['num'], [0, 1, 2, 3, 4]),
    ('cat', orig_p.named_transformers_['cat'], [5, 6, 7, 8])
]
fixed_p.n_features_in_ = 9

# Create pipeline
fixed_pipeline = Pipeline(steps=[
    ('preprocessor', fixed_p),
    ('classifier', classifier)
])

# 3. Save
joblib.dump(fixed_pipeline, FINAL_MODEL_PATH)
print(f"SUCCESS: {FINAL_MODEL_PATH} REGENERATED SUCCESSFULLY.")
