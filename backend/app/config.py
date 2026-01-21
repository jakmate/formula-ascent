from datetime import datetime
import logging
import os
from pathlib import Path

CURRENT_YEAR = datetime.now().year
SEASON_END_MONTH = 12
SEED = 69
NOT_PARTICIPATED_CODES = ["nan", "DNS", "WD", "DNQ", "DNA", "C", "EX"]
RETIREMENT_CODES = ["Ret", "NC", "DSQ", "DSQP"]

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
STATE_FILE = BASE_DIR / "system_state.json"
DATA_DIR = BASE_DIR / "data"
PROFILES_DIR = DATA_DIR / "driver_profiles"
ACADEMIES_DIR = DATA_DIR / "academies"


def get_schedule_dir():
    """Get schedule directory, falling back to previous year if current year is empty"""
    current_year_dir = DATA_DIR / "schedules" / str(CURRENT_YEAR)

    # Check if current year directory exists and has any JSON files
    if current_year_dir.exists():
        json_files = list(current_year_dir.glob("*.json"))
        if json_files:
            return current_year_dir

    # Fallback to previous year
    previous_year_dir = DATA_DIR / "schedules" / str(CURRENT_YEAR - 1)
    if previous_year_dir.exists():
        return previous_year_dir

    # Return current year dir as default (will be created if needed)
    return current_year_dir


SCHEDULE_DIR = get_schedule_dir()

PORT = int(os.environ.get("PORT", 8000))

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)
