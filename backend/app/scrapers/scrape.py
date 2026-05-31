import gc
import logging

from bs4 import BeautifulSoup

from app.config import CURRENT_YEAR
from app.scrapers.championship_scraper import process_championship
from app.scrapers.driver_scraper import scrape_drivers
from app.scrapers.entries_scraper import process_entries
from app.scrapers.qualifying_scraper import scrape_quali
from app.scrapers.schedule_scraper import scrape_schedules
from app.scrapers.scraping_utils import create_session, safe_request

log = logging.getLogger(__name__)

BASE_URL = "https://en.wikipedia.org/wiki/"


def map_url(num, year):
    if num == 1:
        return f"{BASE_URL}{year}_Formula_One_World_Championship"
    if (year <= 2016 and num == 2) or (year <= 2018 and num == 3):
        return f"{BASE_URL}{year}_GP{num}_Series"
    if num == 3:
        return f"{BASE_URL}{year}_FIA_Formula_{num}_Championship"
    if num == 2:
        return f"{BASE_URL}{year}_Formula_{num}_Championship"
    return None


def scrape_wiki(
    session=None,
    formulas=(1, 2, 3),
    start_year=2010,
    end_year=CURRENT_YEAR + 1,
):
    if not session:
        session = create_session()

    for num in formulas:
        for year in range(start_year, end_year):
            url = map_url(num, year)
            log.info("Processing F%s %s...", num, year)

            response = safe_request(session, url)
            if response is None:
                log.info("Skipping F%s %s due to request failure", num, year)
                continue

            try:
                soup = BeautifulSoup(response.text, "lxml")
                response.close()
                del response

                process_entries(soup, year, num)
                process_championship(soup, "Teams'", year, "teams_standings", num)
                process_championship(soup, "Drivers'", year, "drivers_standings", num)
                scrape_quali(soup, year, num)

                soup.decompose()
                gc.collect()

            except Exception:
                log.exception("Error processing data for F%s %s:", num, year)


def scrape():
    session = create_session()
    try:
        scrape_wiki(session)
        scrape_drivers(session)
        scrape_schedules(session)
    finally:
        session.close()


def scrape_current_year():
    session = create_session()
    try:
        scrape_wiki(session, start_year=CURRENT_YEAR)
        scrape_drivers(session)
        scrape_schedules(session)
    finally:
        session.close()


if __name__ == "__main__":  # pragma: no cover
    scrape()
