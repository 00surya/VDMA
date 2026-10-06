"""Load this workspace's private environment without overriding exported settings."""
from pathlib import Path


def load_environment():
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv(Path(__file__).resolve().parents[1] / '.env', override=False)
