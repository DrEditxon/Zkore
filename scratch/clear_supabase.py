import os
import sys
import requests
from dotenv import load_dotenv

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'backend')))

load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))

from app.core.config import settings

def clear_supabase_models():
    url = settings.SUPABASE_URL
    key = settings.SUPABASE_KEY
    if not url or not key:
        print("Supabase credentials not found in env.")
        return

    bucket = "models"
    headers = {
        "apikey": key,
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json"
    }
    
    # List models
    r = requests.post(f"{url}/storage/v1/object/list/{bucket}", headers=headers, json={"prefix": "", "limit": 200})
    if r.status_code != 200:
        print("Failed to list objects:", r.text)
        return
        
    items = r.json()
    if not items:
        print("No models found in Supabase.")
        return
        
    print(f"Found {len(items)} models to delete.")
    
    # Delete models
    for item in items:
        name = item.get("name")
        if name:
            r_del = requests.delete(f"{url}/storage/v1/object/{bucket}/{name}", headers=headers)
            if r_del.status_code in (200, 204):
                print(f"Deleted {name}")
            else:
                print(f"Failed to delete {name}:", r_del.text)

if __name__ == "__main__":
    clear_supabase_models()
