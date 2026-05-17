import json
from unittest.mock import MagicMock, Mock, mock_open, patch

import pandas as pd

from app.scrapers.driver_scraper import (
    extract_dob_from_result,
    extract_nationality_from_result,
    get_all_drivers_from_data,
    get_driver_filename,
    save_profile,
    scrape_drivers,
    search_wikidata_drivers,
)


class TestGetDriverFilename:
    def test_basic_name(self):
        assert get_driver_filename("Lewis Hamilton") == "lewis_hamilton.json"

    def test_multiple_spaces(self):
        assert get_driver_filename("Jean  Eric  Vergne") == "jean_eric_vergne.json"

    def test_hyphens(self):
        assert get_driver_filename("Jean-Eric Vergne") == "jean_eric_vergne.json"


class TestExtractNationalityFromResult:
    def test_with_nationality_label(self):
        result = {"nationalityLabel": {"value": "United Kingdom"}}
        assert extract_nationality_from_result(result) == "United Kingdom"

    def test_with_citizenship_only(self):
        result = {
            "nationalityLabel": {"value": ""},
            "citizenshipLabel": {"value": "France"},
        }
        assert extract_nationality_from_result(result) == "France"

    def test_no_nationality_data(self):
        result = {}
        assert extract_nationality_from_result(result) is None


class TestExtractDobFromResult:
    def test_with_dob(self):
        result = {"dob": {"value": "1985-01-07T00:00:00Z"}}
        assert extract_dob_from_result(result) == "1985-01-07"

    def test_without_dob(self):
        result = {}
        assert extract_dob_from_result(result) is None


class TestSaveProfile:
    def test_save_profile(self):
        mock_file = mock_open()
        mock_path = MagicMock()
        mock_path.open = mock_file
        profile = {
            "name": "Lewis Hamilton",
            "dob": "1985-01-07",
            "nationality": "United Kingdom",
        }

        save_profile(mock_path, profile)

        mock_path.open.assert_called_once_with("w", encoding="utf-8")
        handle = mock_file.return_value.__enter__.return_value
        written_content = "".join(call.args[0] for call in handle.write.call_args_list)
        assert json.loads(written_content) == profile


class TestSearchWikidataDrivers:
    @patch("app.scrapers.driver_scraper.SPARQL_ENDPOINT", "https://test.endpoint")
    def test_successful_batch_query(self):
        mock_session = Mock()
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "results": {
                "bindings": [
                    {
                        "nameMatch": {"value": "Lewis Hamilton"},
                        "person": {"value": "https://wikidata.org/entity/Q1"},
                        "dob": {"value": "1985-01-07T00:00:00Z"},
                        "nationalityLabel": {"value": "United Kingdom"},
                    },
                ],
            },
        }
        mock_session.get.return_value = mock_response

        results = search_wikidata_drivers(["Lewis Hamilton"], mock_session)

        assert "Lewis Hamilton" in results
        assert (
            results["Lewis Hamilton"]["person"]["value"]
            == "https://wikidata.org/entity/Q1"
        )

    @patch("app.scrapers.driver_scraper.SPARQL_ENDPOINT", "https://test.endpoint")
    def test_failed_query(self):
        mock_session = Mock()
        mock_response = Mock()
        mock_response.status_code = 500
        mock_session.get.return_value = mock_response

        results = search_wikidata_drivers(["Lewis Hamilton"], mock_session)

        assert results == {}

    @patch("app.scrapers.driver_scraper.SPARQL_ENDPOINT", "https://test.endpoint")
    def test_query_exception(self):
        mock_session = Mock()
        mock_session.get.side_effect = Exception("Connection error")

        results = search_wikidata_drivers(["Lewis Hamilton"], mock_session)

        assert results == {}

    @patch("app.scrapers.driver_scraper.SPARQL_ENDPOINT", "https://test.endpoint")
    def test_batch_processing(self):
        mock_session = Mock()
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"results": {"bindings": []}}
        mock_session.get.return_value = mock_response

        # Test with 150 drivers to trigger multiple batches
        drivers = [f"Driver{i}" for i in range(150)]
        search_wikidata_drivers(drivers, mock_session, batch_size=100)

        # Should be called twice (batch 0-99, 100-149)
        assert mock_session.get.call_count == 2


class TestGetAllDriversFromData:
    def _make_year_dir(self, name, csv_exists=True):
        year_dir = MagicMock()
        year_dir.name = name
        entries_file = MagicMock()
        entries_file.exists.return_value = csv_exists
        year_dir.__truediv__ = MagicMock(return_value=entries_file)
        return year_dir, entries_file

    @patch("app.scrapers.driver_scraper.pd.read_csv")
    @patch("app.scrapers.driver_scraper.Path")
    def test_extracts_drivers_from_files(self, mock_path, mock_read_csv):
        year_dir, _ = self._make_year_dir("2023")
        mock_path.return_value.__truediv__.return_value.glob.return_value = [year_dir]
        mock_read_csv.return_value = pd.DataFrame(
            {"Driver": ["Lewis Hamilton", "Max Verstappen", "Lewis Hamilton"]}
        )
        drivers = get_all_drivers_from_data()
        assert "Lewis Hamilton" in drivers
        assert "Max Verstappen" in drivers

    @patch("app.scrapers.driver_scraper.Path")
    def test_handles_no_data_dirs(self, mock_path):
        mock_path.return_value.__truediv__.return_value.glob.return_value = []
        drivers = get_all_drivers_from_data()
        assert drivers == []

    @patch("app.scrapers.driver_scraper.pd.read_csv")
    @patch("app.scrapers.driver_scraper.Path")
    def test_handles_missing_driver_column(self, mock_path, mock_read_csv):
        year_dir, _ = self._make_year_dir("2023")
        mock_path.return_value.__truediv__.return_value.glob.return_value = [year_dir]
        mock_read_csv.return_value = pd.DataFrame({"Team": ["Mercedes"]})
        assert get_all_drivers_from_data() == []

    @patch("app.scrapers.driver_scraper.pd.read_csv")
    @patch("app.scrapers.driver_scraper.Path")
    def test_handles_read_error(self, mock_path, mock_read_csv):
        year_dir, _ = self._make_year_dir("2023")
        mock_path.return_value.__truediv__.return_value.glob.return_value = [year_dir]
        mock_read_csv.side_effect = Exception("Read error")
        assert get_all_drivers_from_data() == []

    @patch("app.scrapers.driver_scraper.Path")
    def test_skips_non_digit_year_dirs(self, mock_path):
        bad_dirs = [MagicMock(name=n) for n in ["latest", "abcd"]]
        for d in bad_dirs:
            d.name = d._mock_name  # ensure .name is set
        mock_path.return_value.__truediv__.return_value.glob.return_value = bad_dirs
        assert get_all_drivers_from_data() == []


class TestScrapeDrivers:
    # helper to build a mock PROFILES_DIR
    def _mock_profiles_dir(self, file_exists=False):
        mock_dir = MagicMock()
        mock_file = MagicMock()
        mock_file.exists.return_value = file_exists
        mock_file.open = mock_open(
            read_data=json.dumps(
                {"scraped": True, "dob": "1985-01-07", "nationality": "UK"}
            )
        )
        mock_dir.__truediv__ = MagicMock(return_value=mock_file)
        return mock_dir, mock_file

    @patch("app.scrapers.driver_scraper.create_session")
    @patch("app.scrapers.driver_scraper.get_all_drivers_from_data")
    @patch("app.scrapers.driver_scraper.search_wikidata_drivers")
    @patch("app.scrapers.driver_scraper.save_profile")
    def test_no_drivers_found(
        self, mock_save, mock_search, mock_get_drivers, mock_session
    ):
        mock_get_drivers.return_value = []
        mock_session.return_value = Mock()
        scrape_drivers()
        mock_search.assert_not_called()
        mock_save.assert_not_called()

    @patch("app.scrapers.driver_scraper.create_session")
    @patch("app.scrapers.driver_scraper.get_all_drivers_from_data")
    @patch("app.scrapers.driver_scraper.search_wikidata_drivers")
    @patch("app.scrapers.driver_scraper.save_profile")
    def test_new_driver_no_results(
        self, mock_save, mock_search, mock_get_drivers, mock_session
    ):
        mock_get_drivers.return_value = ["New Driver"]
        mock_search.return_value = {}
        mock_session.return_value = Mock()
        mock_profiles_dir, _ = self._mock_profiles_dir(file_exists=False)

        with patch("app.scrapers.driver_scraper.PROFILES_DIR", mock_profiles_dir):
            scrape_drivers()

        assert mock_save.called
        saved_profile = mock_save.call_args[0][1]
        assert saved_profile["scraped"] is False
        assert saved_profile["name"] == "New Driver"

    @patch("app.scrapers.driver_scraper.create_session")
    @patch("app.scrapers.driver_scraper.get_all_drivers_from_data")
    @patch("app.scrapers.driver_scraper.search_wikidata_drivers")
    @patch("app.scrapers.driver_scraper.save_profile")
    def test_new_driver_with_results(
        self, mock_save, mock_search, mock_get_drivers, mock_session
    ):
        mock_get_drivers.return_value = ["Lewis Hamilton"]
        mock_search.return_value = {
            "Lewis Hamilton": {
                "person": {"value": "https://wikidata.org/entity/Q1"},
                "dob": {"value": "1985-01-07T00:00:00Z"},
                "nationalityLabel": {"value": "United Kingdom"},
            }
        }
        mock_session.return_value = Mock()
        mock_profiles_dir, _ = self._mock_profiles_dir(file_exists=False)

        with patch("app.scrapers.driver_scraper.PROFILES_DIR", mock_profiles_dir):
            scrape_drivers()

        saved_profile = mock_save.call_args[0][1]
        assert saved_profile["scraped"] is True
        assert saved_profile["dob"] == "1985-01-07"
        assert saved_profile["wikidata_id"] == "Q1"

    @patch("app.scrapers.driver_scraper.create_session")
    @patch("app.scrapers.driver_scraper.get_all_drivers_from_data")
    @patch("app.scrapers.driver_scraper.search_wikidata_drivers")
    @patch("app.scrapers.driver_scraper.save_profile")
    def test_existing_driver_needs_update(
        self, mock_save, mock_search, mock_get_drivers, mock_session
    ):
        mock_get_drivers.return_value = ["Lewis Hamilton"]
        mock_search.return_value = {
            "Lewis Hamilton": {
                "person": {"value": "https://wikidata.org/entity/Q1"},
                "dob": {"value": "1985-01-07T00:00:00Z"},
                "nationalityLabel": {"value": "United Kingdom"},
            }
        }
        mock_session.return_value = Mock()
        mock_profiles_dir, mock_file = self._mock_profiles_dir(file_exists=True)
        # existing profile has different nationality → triggers rescrape
        mock_file.open = mock_open(
            read_data=json.dumps(
                {"scraped": True, "dob": "1985-01-07", "nationality": "UK"}
            )
        )

        with patch("app.scrapers.driver_scraper.PROFILES_DIR", mock_profiles_dir):
            scrape_drivers()

        assert mock_save.called

    @patch("app.scrapers.driver_scraper.get_all_drivers_from_data")
    def test_with_provided_session(self, mock_get_drivers):
        mock_get_drivers.return_value = []
        session = Mock()
        scrape_drivers(session=session)
        session.close.assert_not_called()

    @patch("app.scrapers.driver_scraper.create_session")
    @patch("app.scrapers.driver_scraper.get_all_drivers_from_data")
    @patch("app.scrapers.driver_scraper.search_wikidata_drivers")
    @patch(
        "app.scrapers.driver_scraper.DRIVER_ALIASES", {"Test Driver": "Aliased Driver"}
    )
    def test_driver_alias_mapping(self, mock_search, mock_get_drivers, mock_session):
        mock_get_drivers.return_value = ["Test Driver"]
        mock_search.return_value = {}
        mock_session.return_value = Mock()
        mock_profiles_dir, _ = self._mock_profiles_dir(file_exists=False)

        with patch("app.scrapers.driver_scraper.PROFILES_DIR", mock_profiles_dir):
            scrape_drivers()

        call_args = mock_search.call_args[0][0]
        assert "Aliased Driver" in call_args
