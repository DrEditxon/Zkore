import sys
import time
sys.path.append('.')
import traceback
from app.services.history_service import history_service

try:
    start = time.time()
    res = history_service.get_league_history('PD')
    end = time.time()
    print(f"Time taken: {end - start:.2f} seconds")
except Exception as e:
    traceback.print_exc()
