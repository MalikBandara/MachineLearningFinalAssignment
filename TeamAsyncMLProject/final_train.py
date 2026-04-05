import pandas as pd
import joblib
import os
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier

# Paths
ARTIFACTS_DIR = r'd:\MLFINAL\MachineLearningFinalAssignment\TeamAsyncMLProject\artifacts'
DATA_PATH = os.path.join(ARTIFACTS_DIR, 'processed_data.csv')
MODEL_PATH = os.path.join(ARTIFACTS_DIR, 'model.joblib')

print("Starting FINAL NATIVE 1.3.2 TRAINING...")

# 1. Load Data
df = pd.read_csv(DATA_PATH)

# Features from Notebook 1
num_features = ['quantity', 'order_month', 'order_day_of_week', 'order_hour', 'shipping_delay_days']
cat_features = ['category', 'payment_method', 'device_type', 'channel']
target = 'customer_segment'

# Map target to numeric
label_map = {'New': 0, 'Returning': 1, 'VIP': 2}
df[target] = df[target].map(label_map)

# 2. Define Preprocessor using INDICES (Crucial for Vertex AI compatibility)
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), [0, 1, 2, 3, 4]), # Indices for num features
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), [5, 6, 7, 8]) # Indices for cat features
    ])

# 3. Define Final Pipeline
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])

# 4. Fit on the local processed dataset
# We use X.values to ensure the transformer learns indices, not column names
X = df[num_features + cat_features]
y = df[target]

print(f"Fitting model on {len(X)} rows...")
pipeline.fit(X.values, y) 

# 5. Save
joblib.dump(pipeline, MODEL_PATH)
print(f"✅ SUCCESS: NATIVE 1.3.2 model saved to {MODEL_PATH}")
print("Now upload this file to GCS and deploy to Version 1.3 container.")
