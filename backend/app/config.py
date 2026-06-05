import logging
from datetime import UTC, datetime
from pathlib import Path

CURRENT_YEAR = datetime.now(UTC).year
SEASON_END_MONTH = 12
SEED = 69
F2_WEIGHTED_SPRINT_WEIGHT = 0.3
EXPERIENCE_SEASON_PARTICIPATION_THRESHOLD = 0.3
NOT_PARTICIPATED_CODES = ["nan", "DNS", "WD", "DNQ", "DNA", "C", "EX"]
RETIREMENT_CODES = ["Ret", "NC", "DSQ", "DSQP"]

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
STATE_FILE = BASE_DIR / "system_state.json"
DATA_DIR = BASE_DIR / "data"
PROFILES_DIR = DATA_DIR / "driver_profiles"
ACADEMIES_DIR = DATA_DIR / "academies"
SCHEDULE_DIR = DATA_DIR / "schedules" / str(CURRENT_YEAR)

logging.basicConfig(level=logging.INFO)
