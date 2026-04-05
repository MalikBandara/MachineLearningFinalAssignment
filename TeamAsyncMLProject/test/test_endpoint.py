import os
from dotenv import load_dotenv
from google.cloud import aiplatform

# .env loading
script_dir = os.path.dirname(os.path.abspath(__file__))
dotenv_path = os.path.join(script_dir, '.env')
load_dotenv(dotenv_path=dotenv_path)

# New Project Details
PROJECT_ID = "613147120640" 
LOCATION = "asia-southeast1"
ENDPOINT_ID = "2983601767784120320"

# Initialize Vertex AI
print(f"Connecting to NATIVE 1.3 Endpoint: {ENDPOINT_ID} in {PROJECT_ID}")
aiplatform.init(project=PROJECT_ID, location=LOCATION)
endpoint = aiplatform.Endpoint(f"projects/{PROJECT_ID}/locations/{LOCATION}/endpoints/{ENDPOINT_ID}")

# ─── THE FINAL VERIFIED PAYLOAD (NUMPY ARRAY FORMAT) ───
# Order: quantity, month, day, hour, shipping_delay, category, payment_method, device_type, channel
instances = [
    [2.0, 6.0, 3.0, 14.0, 2.5, "Electronics", "Credit Card", "Mobile", "Organic"]
]

print("Sending inference request to Vertex AI...")

try:
    response = endpoint.predict(instances=instances)
    
    print("\n✅ PREDICTION SUCCESSFUL!")
    print("=" * 38)
    
    prediction = response.predictions[0]
    label_map = {0: "New", 1: "Returning", 2: "VIP"}
    segment = label_map.get(int(prediction), str(prediction))
    
    print(f"Prediction Result ID: {prediction}")
    print(f"Customer Segment    : {segment}")
    print("=" * 38)

except Exception as e:
    print(f"❌ ERROR: {e}")
