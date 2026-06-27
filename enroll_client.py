import base64
import sys
import requests

def enroll(name, image_path):
    try:
        with open(image_path, "rb") as f:
            img_b64 = base64.b64encode(f.read()).decode("utf-8")
    except FileNotFoundError:
        print(f"❌ Error: Could not find image at '{image_path}'")
        sys.exit(1)
        
    payload = {
        "name": name,
        "images": [img_b64]
    }
    
    print(f"Sending enrollment request for '{name}'...")
    try:
        response = requests.post("http://localhost:8000/api/v1/enroll", json=payload)
    except requests.exceptions.ConnectionError:
        print("❌ Error: Could not connect to the API. Is the server running? (python -m entrypoints.server)")
        sys.exit(1)
    
    if response.status_code == 200:
        data = response.json()
        if data["success"]:
            print(f"✅ Successfully enrolled '{name}'!")
            print(f"Identity ID: {data['identity_id']}")
            print(f"Embeddings stored: {data['embeddings_stored']}")
        else:
            print(f"❌ Enrollment failed: {data['message']}")
            if data.get("quality_reports"):
                report = data["quality_reports"][0]
                print(f"Reasons: {', '.join(report['rejection_reasons'])}")
    else:
        print(f"❌ HTTP Error {response.status_code}: {response.text}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python enroll_client.py \"Your Name\" path/to/photo.jpg")
        sys.exit(1)
        
    enroll(sys.argv[1], sys.argv[2])
