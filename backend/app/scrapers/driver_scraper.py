import json
import logging
import re
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from app.config import PROFILES_DIR
from app.scrapers.scraping_utils import create_session

log = logging.getLogger(__name__)

SPARQL_ENDPOINT = "https://query.wikidata.org/sparql"
DRIVER_ALIASES = {
    "Lucas di Grassi": "Lucas Di Grassi",
    "Alfonso Celis Jr.": "Alfonso Celis",
    "Keyvan Andres": "Keyvan Andres Soori",
    "Gabriel Bortoleto": "Gabriel Lourenzo Bortoleto Oliveira",
    "Gabriel Chaves": "Gabby Chaves",
    "Robert Kubica": "Robert Jozef Kubica",
    "Yifei Ye": "Ye Yifei",
    "Frederik Vesti": "Frederik Stamm Vesti",
}
DRIVER_OVERRIDES = {
    "Alex García": {
        "wikidata_id": "Q114772333",
        "wikidata_url": "https://www.wikidata.org/wiki/Q114772333",
    },
}
NATIONALITY_TO_COUNTRY = {
    "Austrian-Swiss": "Austria",
    "British": "United Kingdom",
    "Dutch": "Netherlands",
    "French": "France",
    "Indian": "India",
    "Irish": "Ireland",
    "Italian": "Italy",
    "Japanese": "Japan",
    "Mexican": "Mexico",
    "Russian": "Russia",
}


def get_driver_filename(driver_name):
    """Create safe filename from driver name."""
    safe_name = re.sub(r"[^\w\s-]", "", driver_name)
    safe_name = re.sub(r"[-\s]+", "_", safe_name)
    return f"{safe_name.lower()}.json"


def parse_description(description: str) -> dict:
    """Parse nationality and DOB from descriptions.

    E.g. 'Indian racing driver (born 2000)' or
    'British racing driver (born 14 March 1995)'
    """
    result = {}

    # Extract nationality - word(s) before "racing driver"
    nationality_match = re.match(
        r"^([A-Z][a-z]+(?:[- ][A-Za-z]+)*)\s+racing driver", description
    )
    if nationality_match:
        adj = nationality_match.group(1)
        result["nationality"] = NATIONALITY_TO_COUNTRY.get(adj, adj)

    # Extract DOB - handles "born 2000" or "born 14 March 1995" or "born March 14, 1995"
    dob_match = re.search(
        r"born\s+(\d{1,2}\s+\w+\s+\d{4}|\w+\s+\d{1,2},?\s+\d{4}|\d{4})", description
    )
    if dob_match:
        raw = dob_match.group(1).strip().rstrip(",")
        for fmt in ("%d %B %Y", "%B %d %Y", "%Y"):
            try:
                result["dob"] = (
                    datetime.strptime(raw, fmt)
                    .replace(tzinfo=UTC)
                    .strftime("%Y-%m-%d" if fmt != "%Y" else "%Y-01-01")
                )
                break
            except ValueError:
                continue

    return result


def fetch_driver_by_qid(qid, session):
    """Fetch driver bio data from Wikidata by QID."""
    query = f"""
    SELECT ?person ?personLabel ?dob ?nationality ?nationalityLabel
        ?citizenship ?citizenshipLabel ?description WHERE {{
    BIND(wd:{qid} AS ?person)
    OPTIONAL {{ ?person wdt:P569 ?dob }}
    OPTIONAL {{ ?person wdt:P1532 ?nationality }}
    OPTIONAL {{ ?person wdt:P27 ?citizenship }}
    OPTIONAL {{ ?person schema:description ?description .
                FILTER(LANG(?description) = "en") }}
    SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en" }}
    }}
    """
    try:
        response = session.get(
            SPARQL_ENDPOINT,
            params={"query": query, "format": "json"},
            timeout=30,
        )
        if response.status_code != 200:
            return None

        bindings = response.json()["results"]["bindings"]
        if not bindings:
            return None

        # Merge all rows (multiple citizenship/nationality rows possible)
        merged = {
            **bindings[0],
            "_all_nationalityLabels": set(),
            "_all_citizenshipLabels": set(),
        }
        for binding in bindings:
            if "nationalityLabel" in binding and binding["nationalityLabel"]["value"]:
                merged["_all_nationalityLabels"].add(
                    binding["nationalityLabel"]["value"]
                )
            if "citizenshipLabel" in binding and binding["citizenshipLabel"]["value"]:
                merged["_all_citizenshipLabels"].add(
                    binding["citizenshipLabel"]["value"]
                )
        return merged

    except Exception:
        log.exception("QID fetch error for %s:", qid)
        return None


def search_wikidata_drivers(driver_names, session, batch_size=100):
    """Search for racing drivers in Wikidata using batched SPARQL."""
    results = {}

    # Process in batches
    for i in range(0, len(driver_names), batch_size):
        batch = driver_names[i : i + batch_size]

        # Build VALUES clause for batch
        values_list = []
        for name in batch:
            values_list.extend([f'"{name}"@en', f'"{name}"@mul'])
        values_clause = " ".join(values_list)

        query = f"""
        SELECT ?person ?personLabel ?dob ?nationality ?nationalityLabel
            ?citizenship ?citizenshipLabel ?nameMatch ?description WHERE {{
        VALUES ?nameMatch {{ {values_clause} }}
        {{
            ?person rdfs:label ?nameMatch .
        }} UNION {{
            ?person skos:altLabel ?nameMatch .
        }}
        ?person wdt:P106 ?occupation .
        FILTER(?occupation = wd:Q10349745 ||
            ?occupation = wd:Q378622 || ?occupation = wd:Q10841764)

        OPTIONAL {{ ?person wdt:P569 ?dob }}
        OPTIONAL {{ ?person wdt:P1532 ?nationality }}
        OPTIONAL {{ ?person wdt:P27 ?citizenship }}
        OPTIONAL {{ ?person schema:description ?description .
                    FILTER(LANG(?description) = "en") }}

        SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en" }}
        }}
        """

        try:
            response = session.get(
                SPARQL_ENDPOINT,
                params={"query": query, "format": "json"},
                timeout=30,
            )

            if response.status_code == 200:
                data = response.json()
                # Group results by matched name
                for binding in data["results"]["bindings"]:
                    name = binding["nameMatch"]["value"]
                    if name not in results:
                        # First row: store base fields, init citizenship lists
                        results[name] = {
                            **binding,
                            "_all_nationalityLabels": set(),
                            "_all_citizenshipLabels": set(),
                        }
                    # Accumulate every citizenship/nationality row
                    merged = results[name]
                    if (
                        "nationalityLabel" in binding
                        and binding["nationalityLabel"]["value"]
                    ):
                        merged["_all_nationalityLabels"].add(
                            binding["nationalityLabel"]["value"]
                        )
                    if (
                        "citizenshipLabel" in binding
                        and binding["citizenshipLabel"]["value"]
                    ):
                        merged["_all_citizenshipLabels"].add(
                            binding["citizenshipLabel"]["value"]
                        )
            else:
                log.warning(response.status_code)

        except Exception:
            log.exception("Batch query error for batch %s:", i // batch_size + 1)

    return results


def extract_all_nationalities_from_result(result):
    """Return all citizenship/nationality values collected across all SPARQL rows."""
    values = set()
    values.update(result.get("_all_nationalityLabels", set()))
    values.update(result.get("_all_citizenshipLabels", set()))
    if "description" in result and result["description"]["value"]:
        parsed = parse_description(result["description"]["value"])
        if parsed.get("nationality"):
            values.add(parsed["nationality"])
    return values


def extract_nationality_from_result(result):
    """Extract single nationality from result, preferring country for sport.

    When multiple values exist picks first nationalityLabel, else citizenshipLabel.
    """
    all_nat = result.get("_all_nationalityLabels", set())
    if all_nat:
        return next(iter(all_nat))
    all_cit = result.get("_all_citizenshipLabels", set())
    if all_cit:
        return next(iter(all_cit))
    if "description" in result and result["description"]["value"]:
        parsed = parse_description(result["description"]["value"])
        if parsed.get("nationality"):
            return parsed["nationality"]
    return None


def extract_dob_from_result(result):
    """Extract date of birth from Wikidata result."""
    if "dob" in result and result["dob"]["value"]:
        # Wikidata returns dates in ISO format, extract just the date part
        return result["dob"]["value"].split("T")[0]
    if "description" in result and result["description"]["value"]:
        parsed = parse_description(result["description"]["value"])
        if parsed.get("dob"):
            return parsed["dob"]
    return None


def save_profile(filename, profile):
    """Save profile to JSON file."""
    with filename.open("w", encoding="utf-8") as f:
        json.dump(profile, f, indent=2, ensure_ascii=False)


def needs_rescrape(driver, existing_profile, new_data, session=None):
    """Check if profile needs to be rescraped based on data changes."""
    if not existing_profile.get("scraped", False):
        # If previous scrape failed, try again
        return True

    override = DRIVER_OVERRIDES.get(driver, {})
    if override.get("wikidata_id") and session:
        new_data = fetch_driver_by_qid(override["wikidata_id"], session) or new_data

    # Extract new values
    new_dob = extract_dob_from_result(new_data)
    dob_changed = existing_profile.get("dob") != new_dob

    existing_nationality = existing_profile.get("nationality")
    new_nationality = extract_nationality_from_result(new_data)

    if existing_nationality != new_nationality:
        new_nationalities = extract_all_nationalities_from_result(new_data)
        if len(new_nationalities) > 1 and existing_nationality in new_nationalities:
            log.info("Nationality ambigious for %s", driver)
            log.info("Existing nationality: %s", existing_nationality)
            log.info("Wikidata returned: %s", new_nationalities)
            nationality_changed = False
        else:
            nationality_changed = True
    else:
        nationality_changed = False

    return dob_changed or nationality_changed


def get_all_drivers_from_data():
    """Extract all driver names from data files."""
    all_drivers = set()
    series_map = {
        "F1": "f1_{year}_entries.csv",
        "F2": "f2_{year}_entries.csv",
        "F3": "f3_{year}_entries.csv",
    }

    for series, pattern in series_map.items():
        for year_dir in (Path("data") / series).glob("*"):
            year = year_dir.name
            if not year.isdigit():
                continue

            entries_file = Path(year_dir) / pattern.format(year=year)

            if entries_file.exists():
                try:
                    df = pd.read_csv(entries_file)
                    if "Driver" in df.columns:
                        drivers = df["Driver"].dropna().str.strip().unique()
                        all_drivers.update(drivers)
                except Exception:
                    log.exception("Error reading %s", entries_file)

    return sorted(all_drivers)


def build_profile_from_result(driver, result, session=None):
    """Build a profile dict from a Wikidata result, applying hard-coded overrides."""
    override = DRIVER_OVERRIDES.get(driver, {})
    wikidata_id = (
        override.get("wikidata_id") or result["person"]["value"].split("/")[-1]
    )
    wikidata_url = override.get("wikidata_url") or result["person"]["value"]

    # If override QID differs from SPARQL result, fetch correct entity for bio data
    bio_result = result
    if override.get("wikidata_id") and session:
        correct_result = fetch_driver_by_qid(override["wikidata_id"], session)
        if correct_result:
            bio_result = correct_result

    return {
        "name": driver,
        "dob": extract_dob_from_result(bio_result),
        "nationality": extract_nationality_from_result(bio_result),
        "wikidata_id": wikidata_id,
        "wikidata_url": wikidata_url,
        "scraped": True,
        "scraped_date": datetime.now(UTC).isoformat(),
    }


def scrape_drivers(session=None):
    """Main function to scrape all driver profiles using batched queries."""
    if not session:
        session = create_session()

    log.info("Scanning data files for driver names...")
    all_drivers = get_all_drivers_from_data()

    if not all_drivers:
        log.warning("No drivers found.")
        return

    log.info("Found %s unique drivers", len(all_drivers))

    # Ensure profiles directory exists
    PROFILES_DIR.mkdir(parents=True, exist_ok=True)

    try:
        # Create mapping of original name -> search name
        driver_search_map = {}
        for driver in all_drivers:
            search_name = DRIVER_ALIASES.get(driver, driver)
            if search_name is not None:  # Skip explicitly invalid drivers
                driver_search_map[driver] = search_name

        # Separate drivers into new and existing
        new_drivers = []
        existing_drivers = []

        for driver in driver_search_map:
            profile_file = PROFILES_DIR / get_driver_filename(driver)
            if profile_file.exists():
                existing_drivers.append(driver)
            else:
                new_drivers.append(driver)

        log.info("New drivers:%s, Existing:%s", len(new_drivers), len(existing_drivers))

        # Batch query for new drivers (use search names)
        if new_drivers:
            log.info("Querying %s new drivers in batches...", len(new_drivers))
            search_names = [driver_search_map[d] for d in new_drivers]
            new_results = search_wikidata_drivers(search_names, session)

            # Map results back to original names
            new_results_mapped = {}
            for driver in new_drivers:
                search_name = driver_search_map[driver]
                if search_name in new_results:
                    new_results_mapped[driver] = new_results[search_name]
            new_results = new_results_mapped

            for driver in new_drivers:
                result = new_results.get(driver)
                override = DRIVER_OVERRIDES.get(driver, {})

                if not result and not override:
                    log.warning("No results for %s", driver)
                    profile = {
                        "name": driver,
                        "dob": None,
                        "nationality": None,
                        "scraped": False,
                    }
                elif not result and override:
                    log.warning("No Wikidata result for %s, applying override", driver)
                    bio_result = fetch_driver_by_qid(override["wikidata_id"], session)
                    profile = {
                        "name": driver,
                        "dob": extract_dob_from_result(bio_result)
                        if bio_result
                        else None,
                        "nationality": extract_nationality_from_result(bio_result)
                        if bio_result
                        else None,
                        "wikidata_id": override["wikidata_id"],
                        "wikidata_url": override["wikidata_url"],
                        "scraped": bool(bio_result),
                        "scraped_date": datetime.now(UTC).isoformat(),
                    }
                else:
                    profile = build_profile_from_result(driver, result, session)

                profile_file = PROFILES_DIR / get_driver_filename(driver)
                save_profile(profile_file, profile)

        # Check existing drivers for updates
        if existing_drivers:
            log.info("Checking %s existing drivers for updates", len(existing_drivers))
            search_names = [driver_search_map[d] for d in existing_drivers]
            existing_results = search_wikidata_drivers(search_names, session)

            # Map results back to original names
            existing_results_mapped = {}
            for driver in existing_drivers:
                search_name = driver_search_map[driver]
                if search_name in existing_results:
                    existing_results_mapped[driver] = existing_results[search_name]
            existing_results = existing_results_mapped

            updated_count = 0
            for driver in existing_drivers:
                profile_file = PROFILES_DIR / get_driver_filename(driver)
                with profile_file.open(encoding="utf-8") as f:
                    existing_profile = json.load(f)

                result = existing_results.get(driver)
                if result and needs_rescrape(driver, existing_profile, result, session):
                    log.info("Updating %s...", driver)
                    profile = build_profile_from_result(driver, result, session)
                    save_profile(profile_file, profile)
                    updated_count += 1

            log.info("Updated %s profiles", updated_count)

        log.info("Scraping complete")

    finally:
        session.close()


if __name__ == "__main__":  # pragma: no cover
    scrape_drivers()
