import json
import os
from datetime import datetime

import aiofiles
import pytz
from fastapi import HTTPException

from app.config import SCHEDULE_DIR
from app.models.schedule import ScheduleRequest


class ScheduleService:
    async def get_series_schedule(self, request: ScheduleRequest):
        """Get schedule for a specific racing series with timezone conversion"""
        schedule = await self._open_schedule_file(request)

        user_timezone = request.get_timezone()
        if user_timezone != "UTC":
            schedule = self._convert_schedule_timezone(schedule, user_timezone)

        return schedule

    async def get_next_race(self, request: ScheduleRequest):
        """Get the next upcoming race. If none, return last race of season."""
        schedule = await self._open_schedule_file(request)
        now = datetime.now(pytz.UTC)

        next_race = self._find_next_race(schedule, now)

        if not next_race:
            next_race = schedule[-1].copy() if schedule else None
            if next_race:
                next_race["seasonCompleted"] = True

        if next_race:
            next_race["totalRounds"] = len(schedule)
            user_tz = request.get_timezone()
            if user_tz != "UTC":
                next_race = self._convert_race_timezone(next_race, user_tz)

        return next_race

    def _find_next_race(self, schedule, now):
        """Find the next race with future sessions"""
        upcoming_races = []

        for race in schedule:
            future_sessions = self._get_future_sessions(race, now)
            if future_sessions:
                earliest = min(s["datetime"] for s in future_sessions)
                upcoming_races.append((race, earliest, future_sessions[0]))

        if not upcoming_races:
            return None

        upcoming_races.sort(key=lambda x: x[1])
        race, _, next_session = upcoming_races[0]

        result = race.copy()
        result["nextSession"] = {
            "name": next_session["name"],
            "date": next_session["date_str"],
            "isTBC": next_session["is_tbc"],
        }
        result["seasonCompleted"] = False
        return result

    def _get_future_sessions(self, race, now):
        """Get all future sessions for a race"""
        future = []
        for name, info in race["sessions"].items():
            if start_str := info.get("start"):
                try:
                    dt = self._parse_datetime(start_str)
                    if dt > now:
                        future.append(
                            {
                                "name": name,
                                "datetime": dt,
                                "date_str": start_str,
                                "is_tbc": info.get("time") == "TBC",
                            }
                        )
                except (ValueError, TypeError):
                    continue
        return sorted(future, key=lambda x: x["datetime"])

    def _parse_datetime(self, date_string: str) -> datetime:
        """Parse a date string that could be either YYYY-MM-DD or full ISO format"""
        if len(date_string) == 10:
            return datetime.strptime(date_string, "%Y-%m-%d").replace(tzinfo=pytz.UTC)
        else:
            return self._parse_iso_and_localize(date_string)

    def _parse_iso_and_localize(self, string: str) -> datetime:
        """Parse an ISO timestamp, if naive localize as UTC"""
        dt = datetime.fromisoformat(string)
        if dt.tzinfo is None:
            dt = pytz.UTC.localize(dt)
        return dt

    def _convert_schedule_timezone(self, schedule, target_timezone):
        """Convert all datetime strings in schedule from UTC to target timezone"""
        target_tz = pytz.timezone(target_timezone)

        for race in schedule:
            for _, session_info in race["sessions"].items():
                if session_info.get("time") == "TBC":
                    continue

                for time_field in ["start", "end"]:
                    time_str = session_info.get(time_field)
                    if time_str:
                        utc_dt = self._parse_iso_and_localize(time_str)
                        local_dt = utc_dt.astimezone(target_tz)
                        session_info[time_field] = local_dt.isoformat()

        return schedule

    def _convert_race_timezone(self, race, target_timezone):
        """Convert datetime strings in a single race from UTC to target timezone"""
        target_tz = pytz.timezone(target_timezone)

        for _, session_info in race["sessions"].items():
            if session_info.get("time") == "TBC":
                continue

            for time_field in ["start", "end"]:
                time_str = session_info.get(time_field)
                if time_str and len(time_str) > 10:
                    utc_dt = self._parse_iso_and_localize(time_str)
                    local_dt = utc_dt.astimezone(target_tz)
                    session_info[time_field] = local_dt.isoformat()

        if "nextSession" in race:
            start_str = race["nextSession"].get("date")
            if start_str and len(start_str) > 10:
                utc_dt = self._parse_iso_and_localize(start_str)
                local_dt = utc_dt.astimezone(target_tz)
                race["nextSession"]["date"] = local_dt.isoformat()

        return race

    def _get_schedule_dir(self):
        """Get schedule directory, falling back to previous year if current year is empty"""
        # Check if current year directory exists and has any JSON files
        if SCHEDULE_DIR.exists():
            json_files = list(SCHEDULE_DIR.glob("*.json"))
            if json_files:
                return SCHEDULE_DIR

        # Fallback to previous year
        year_str = SCHEDULE_DIR.name  # Get just the directory name
        if year_str.isdigit():
            prev_year_str = str(int(year_str) - 1)
            previous_year_dir = SCHEDULE_DIR.parent / prev_year_str
        else:
            # fallback if format is unexpected
            previous_year_dir = SCHEDULE_DIR

        if previous_year_dir.exists():
            return previous_year_dir

        # Return current year dir as default
        return SCHEDULE_DIR

    async def _open_schedule_file(self, request: ScheduleRequest):
        file_path = os.path.join(self._get_schedule_dir(), f"{request.series}.json")
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="Schedule data not found")

        async with aiofiles.open(file_path, "r") as f:
            return json.loads(await f.read())
