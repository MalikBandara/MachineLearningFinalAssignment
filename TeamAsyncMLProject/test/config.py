import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Extract Google Cloud configuration variables
GCP_PROJECT_ID = os.getenv("GCP_PROJECT_ID")
GCP_REGION = os.getenv("GCP_REGION")
ENDPOINT_DISPLAY_NAME = os.getenv("ENDPOINT_DISPLAY_NAME")

if not all([GCP_PROJECT_ID, GCP_REGION, ENDPOINT_DISPLAY_NAME]):
    print("Warning: One or more environment variables are missing. Please check your .env file.")
