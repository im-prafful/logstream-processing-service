import boto3
import base64
import subprocess
import os

# --- CONFIGURATION ---
# Replace with your actual ECR repository URI and region
# Note: ECR_REPOSITORY_URI is technically not needed here, but kept for context/consistency
ECR_REPOSITORY_URI = "031415497613.dkr.ecr.ap-south-1.amazonaws.com/logstream-processing-service"
REGION = "ap-south-1" 
TAG = "latest"
# --- END CONFIGURATION ---

def authenticate_docker_to_ecr(repo_uri: str, region: str):
    """Authenticates the local Docker client to ECR using Boto3, but does NOT push."""
    
    print(f"--- 1. Retrieving ECR Auth Token for {region} ---")
    try:
        # Initialize ECR Client
        ecr_client = boto3.client('ecr', region_name=region)
        
        # Get the authorization token
        response = ecr_client.get_authorization_token()
        
        # The token data contains the registry URL and the base64 encoded token
        token_data = response['authorizationData'][0]
        encoded_token = token_data['authorizationToken']
        registry_url = token_data['proxyEndpoint']

        # Decode the token (it yields 'AWS:<password>')
        decoded_token = base64.b64decode(encoded_token).decode()
        username, password = decoded_token.split(':')

    except Exception as e:
        print(f"❌ Error retrieving ECR credentials: {e}")
        return

    print("--- 2. Executing Docker Login (Using Auth Token) ---")
    
    # Construct the docker login command
    login_command = ['docker', 'login', '-u', username, '--password-stdin', registry_url]

    try:
        # Run the login command, feeding the password via standard input
        subprocess.run(
            login_command, 
            input=password, 
            encoding='utf-8', 
            check=True, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE
        )
        print("✅ Docker login successful! You are now authenticated.")
        print(f"   You can manually push using: docker push {repo_uri}:{TAG}")
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Docker login failed. Error: {e.stderr.strip()}")
        return
        
    # --- PUSH SECTION HAS BEEN REMOVED HERE ---
    
# Run the deployment function
if __name__ == "__main__":
    authenticate_docker_to_ecr(ECR_REPOSITORY_URI, REGION)