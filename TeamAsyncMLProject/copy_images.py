import os
import shutil

# Source and destination paths
src_dir = r'C:\Users\Malik Bandara\.gemini\antigravity\brain\bc51b56c-73ba-468e-be80-6904bc730312'
dest_dir = r'd:\MLFINAL\MachineLearningFinalAssignment\TeamAsyncMLProject\report_screenshots'

if not os.path.exists(dest_dir):
    os.makedirs(dest_dir)

print(f"Copying files from {src_dir} to {dest_dir}...")

# Filter for the report images I generated earlier
# The files have timestamp prefixes
files_to_copy = {
    'notebook_preprocessing_ui': '01_preprocessing.png',
    'notebook_pipeline_ui': '02_pipeline.png',
    'prediction_success_ui': '03_final_success.png',
    'mlflow_experiment_ui': '04_mlflow_appendix.png'
}

count = 0
for filename in os.listdir(src_dir):
    for key, new_name in files_to_copy.items():
        if filename.startswith(key) and filename.endswith('.png'):
            src_path = os.path.join(src_dir, filename)
            dest_path = os.path.join(dest_dir, new_name)
            shutil.copy2(src_path, dest_path)
            print(f"Copied: {new_name}")
            count += 1

if count == 0:
    print("No files found to copy. Checking brain directory...")
else:
    print(f"Successfully copied {count} report screenshots.")
