import csv
import os

from bs4 import BeautifulSoup, SoupStrainer

from app.config import CURRENT_YEAR, DATA_DIR
from app.scrapers.scraping_utils import (
    create_session,
    remove_superscripts,
    safe_request,
)


def extract_team_links(soup):
    """Extract team links from the Formula One driver development programs table"""
    f1_heading = soup.find("h3", {"id": "Formula_One"})
    if not f1_heading:
        print("No Formula One section found")
        return []

    table = f1_heading.find_next("table", {"class": "wikitable"})
    if not table:
        print("No driver development programs table found")
        return []

    team_links = []
    rows = table.find_all("tr")
    for row in rows[1:]:
        cells = row.find_all("td")
        if len(cells) < 2:
            continue

        link = cells[0].find("a")
        if not link:
            continue

        href = link.get("href")
        if href and href.startswith("/wiki/"):
            team_links.append(
                {
                    "name": remove_superscripts(link, False),
                    "url": "https://en.wikipedia.org" + href,
                }
            )

    return team_links


def parse_year_range(year_str):
    """Parse year range string and return list of years"""
    year_str = year_str.strip()
    years = []

    # Split on commas
    for segment in year_str.split(","):
        segment = segment.strip()
        if "–" in segment:
            start, end = segment.split("–", 1)
            start = start.strip()
            end = end.strip()

            try:
                start_year = int(start)
                end_year = int(end) if end else CURRENT_YEAR
                years.extend(str(y) for y in range(start_year, end_year + 1))
            except ValueError:
                years.append(segment)
        else:
            years.append(segment)

    return years


def expand_years_in_data(headers, data_rows):
    """Expand year ranges into separate rows and keep only driver and year columns"""
    if not data_rows or not headers:
        return ["Driver", "Year"], []

    # Find year column index
    year_col_idx = None
    year_keywords = {"years", "year", "began", "since"}
    for i, h in enumerate(headers):
        if h.lower() in year_keywords:
            year_col_idx = i
            break

    # If still no year column found, return empty data
    if year_col_idx is None:
        return ["Driver", "Year"], []

    # Expand rows based on year ranges
    driver_col_idx = 0
    expanded_rows = []
    for row in data_rows:
        if driver_col_idx >= len(row) or year_col_idx >= len(row):
            continue

        driver_name = row[driver_col_idx].strip()
        if not driver_name:
            continue

        year_value = row[year_col_idx].strip()
        years = parse_year_range(year_value)

        for year in years:
            expanded_rows.append([driver_name, year])

    return ["Driver", "Year"], expanded_rows


def extract_table_data(table, table_type):
    """Extract data from a table"""
    if not table:
        return None

    all_rows = table.find_all("tr")
    if len(all_rows) < 2:
        return None

    # Extract headers
    header_row = all_rows[0]
    headers = []
    for th in header_row.find_all("th"):
        colspan = int(th.get("colspan", 1))
        if colspan > 1:
            headers.extend([None] * colspan)
        else:
            headers.append(remove_superscripts(th))

    # Check if there's a second header row
    if table_type == "f1_graduates":
        second_row = all_rows[1].find_all("th")
        h2_iter = iter(second_row)
        for i, header in enumerate(headers):
            if header is None:
                try:
                    headers[i] = remove_superscripts(next(h2_iter))
                except StopIteration:
                    break
        data_start = 2
    else:
        data_start = 1

    # Extract data rows
    data_rows = []
    for row in all_rows[data_start:]:
        cells = row.find_all(["td", "th"])
        if cells:
            data_rows.append([remove_superscripts(cell, False) for cell in cells])

    # Expand years and filter to only driver and year columns
    headers, data_rows = expand_years_in_data(headers, data_rows)

    return {"headers": headers, "data": data_rows}


def scrape_academy_page(academy_url, academy_name, session):
    """Scrape data from an academy page"""
    try:
        response = safe_request(session, academy_url)
        if response is None:
            print(f"Failed to fetch {academy_url}")
            return None

        parse_only = SoupStrainer(["h2", "h3", "table"])
        soup = BeautifulSoup(response.text, "lxml", parse_only=parse_only)

        response.close()
        del response

        results = {
            "name": academy_name,
            "url": academy_url,
            "current_drivers": [],
            "former_drivers": [],
            "f1_graduates": [],
        }

        # Find various heading variations
        headings_to_check = [
            ("Current_drivers", "current_drivers"),
            ("Graduates_to_Formula_1", "f1_graduates"),
            ("Graduates_to_Formula_One", "f1_graduates"),
            ("Graduates_to_Red_Bull_Racing_in_Formula_1", "f1_graduates"),
            ("Graduates_to_Toro_Rosso/AlphaTauri/RB", "f1_graduates"),
            ("Former_drivers", "former_drivers"),
            ("Driver_development_programme", None),  # Skip this one
        ]

        team_keywords = {"alpine", "renault", "lotus"}

        for heading_id, key in headings_to_check:
            heading = soup.find("h3", {"id": heading_id}) or soup.find(
                "h2", {"id": heading_id}
            )

            if heading and key:
                # Special handling for Former_drivers with team subheadings
                if key == "former_drivers":
                    # Collect team-specific h3 tables in single pass
                    team_tables = []
                    current = heading.find_next_sibling()

                    # Look ahead to see if there are team-specific h3 headings
                    while current and current.name != "h2":
                        if current.name == "h3":
                            heading_text = current.get_text().lower()
                            if any(kw in heading_text for kw in team_keywords):
                                table = current.find_next_sibling(
                                    "table", {"class": "wikitable"}
                                )
                                if table:
                                    table_data = extract_table_data(table, key)
                                    if table_data:
                                        team_tables.append(table_data)
                        current = current.find_next_sibling()

                    if team_tables:
                        results[key].extend(team_tables)
                    else:
                        # Standard behavior
                        table = heading.find_next("table", {"class": "wikitable"})
                        if table:
                            table_data = extract_table_data(table, key)
                            if table_data:
                                results[key].append(table_data)
                else:
                    # Standard behavior - just get the next table
                    table = heading.find_next("table", {"class": "wikitable"})
                    if table:
                        table_data = extract_table_data(table, key)
                        if table_data:
                            results[key].append(table_data)

        if not any(
            [
                results["current_drivers"],
                results["former_drivers"],
                results["f1_graduates"],
            ]
        ):
            generic_heading = soup.find("h2", {"id": "Driver_development_program"})
            if generic_heading:
                table = generic_heading.find_next("table", {"class": "wikitable"})
                if table:
                    table_data = extract_table_data(table, "current_drivers")
                    if table_data:
                        results["current_drivers"].append(table_data)

        soup.decompose()

        return results

    except Exception as e:
        print(f"Error scraping {academy_url}: {str(e)}")
        return None


def save_academy_data(academy_data, academies_dir):
    """Save academy data to CSV files"""
    if not academy_data:
        return

    os.makedirs(academies_dir, exist_ok=True)

    # Clean name for filename
    safe_name = academy_data["name"].replace(" ", "_").replace("/", "_")
    filepath = os.path.join(academies_dir, f"{safe_name}_drivers.csv")

    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Driver", "Year"])
        # Save current drivers
        for table_data in academy_data["current_drivers"]:
            writer.writerows(table_data["data"])

        # Save former drivers
        for table_data in academy_data["former_drivers"]:
            writer.writerows(table_data["data"])

        # Save F1 graduates
        for table_data in academy_data["f1_graduates"]:
            writer.writerows(table_data["data"])


def scrape_academies(session=None):
    """Scrape driver development program data"""
    if session is None:
        session = create_session()

    academies_dir = os.path.join(DATA_DIR, "academies")

    url = "https://en.wikipedia.org/wiki/Driver_development_program"

    response = safe_request(session, url)
    if response is None:
        print(f"Failed to fetch {url}")
        return None

    parse_only = SoupStrainer(["h3", "table"])
    soup = BeautifulSoup(response.text, "lxml", parse_only=parse_only)

    team_links = extract_team_links(soup)
    if not team_links:
        print("No team links found")
        return None

    print(f"Found {len(team_links)} driver development programs")

    response.close()
    del response
    soup.decompose()

    # Skip Marussia
    skip_programs = {"Marussia F1 Team Young Driver Program"}

    for team in team_links:
        if team["name"] in skip_programs:
            print(f"Skipping {team['name']}")
            continue

        print(f"Scraping {team['name']}...")
        academy_data = scrape_academy_page(team["url"], team["name"], session)

        if academy_data:
            save_academy_data(academy_data, academies_dir)


if __name__ == "__main__":  # pragma: no cover
    scrape_academies()
