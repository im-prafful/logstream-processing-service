import json
import boto3
import os
from dotenv import load_dotenv

# Load AWS credentials from .env if running locally
load_dotenv()

def send_dummy_event():
    print("Connecting to EventBridge in ap-south-1...")
    client = boto3.client('events', region_name='ap-south-1')
    
    # Fake payload that perfectly matches the real anomaly payload
    detail = {
        "cluster_id": 9999,
        "reason": "MANUAL DUMMY TEST for EventBridge Trigger",
        "sample_logs": [
            "[TEST] ERROR: Simulated connection timeout in Database",
            "[TEST] WARN: High latency detected on simulated API endpoint",
            "[TEST] INFO: This is a dummy test log from the Python script"
        ]
    }
    
    print("Payload constructed. Sending event to Logstream-alert-bus...")
    try:
        response = client.put_events(
            Entries=[
                {
                    'Source': 'com.logstream.processing',
                    'DetailType': 'VolumeAnomalyDetected',
                    'Detail': json.dumps(detail),
                    'EventBusName': 'Logstream-alert-bus'
                }
            ]
        )
        print("\n[SUCCESS] Dummy Event sent successfully!")
        print(f"Response from AWS: {json.dumps(response, indent=2)}")
        print("\nCheck the CloudWatch logs for your Publisher Lambda to verify receipt!")
    except Exception as e:
        print(f"\n[ERROR] Failed to send event: {e}")

if __name__ == "__main__":
    send_dummy_event()
