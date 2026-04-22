import requests
import os

key = "ffcdab306fmshe3b6e9b4cd55ed4p137d4ejsn3d598ba17cff"
host = "api-football-v1.p.rapidapi.com"

headers = {
    "x-rapidapi-key": key,
    "x-rapidapi-host": host
}

url = f"https://{host}/v3/timezone"
response = requests.get(url, headers=headers)
print(f"Status: {response.status_code}")
print(f"Body: {response.text}")
