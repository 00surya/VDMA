"""Run with python -m vmd from the project directory."""
import faulthandler
import uvicorn
from .config import load_environment

load_environment()

if __name__ == "__main__":
    faulthandler.enable()
    print("VMD Shield dashboard: http://127.0.0.1:8765", flush=True)
    uvicorn.run("vmd.api:app", host="127.0.0.1", port=8765, access_log=False)
