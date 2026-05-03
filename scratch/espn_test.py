import requests
import json

url = 'https://site.api.espn.com/apis/site/v2/sports/soccer/eng.1/scoreboard'
r = requests.get(url)
print(f"Status: {r.status_code}")
data = r.json()

print(f"Top-level keys: {list(data.keys())}")
if 'events' in data and len(data['events']) > 0:
    ev = data['events'][0]
    print(f"Event name: {ev.get('name')}")
    print(f"Event status: {ev.get('status', {}).get('type', {}).get('description')}")
    
    comp = ev['competitions'][0]
    print(f"Odds available: {'odds' in comp}")
    if 'odds' in comp:
        print(f"Odds details: {json.dumps(comp['odds'][0], indent=2)}")
        
    print(f"Competitors available: {'competitors' in comp}")
    if 'competitors' in comp:
        for c in comp['competitors']:
            print(f"Team: {c.get('team', {}).get('name')}, Score: {c.get('score')}")
