import os
import sys
import uvicorn

# Ensure this directory is on sys.path so imports work when running the script directly
BASE_DIR = os.path.dirname(__file__)
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from api import app

if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)
