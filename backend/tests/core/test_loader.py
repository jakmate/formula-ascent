import pandas as pd
import json
from unittest.mock import patch, mock_open, MagicMock

from app.core.loader import (
    get_file_pattern,
    get_series_directories,
    load_all_entries_data,
    load_year_data,
    load_standings_data,
    load_qualifying_data,
    get_driver_filename,
    load_driver_data,
    merge_entries,
    parse_round_count,
    merge_team_data,
    load_data,
)


class TestGetFilePattern:
    def test_drivers_pattern(self):
        result = get_file_pattern("drivers", "F3", "2023")
        assert result == "f3_2023_drivers_standings.csv"

    def test_entries_pattern(self):
        result = get_file_pattern("entries", "F2", "2022")
        assert result == "f2_2022_entries.csv"

    def test_teams_pattern(self):
        result = get_file_pattern("teams", "F3", "2021")
        assert result == "f3_2021_teams_standings.csv"

    def test_qualifying_pattern_with_round(self):
        result = get_file_pattern("qualifying", "F2", "2023", round_num=5)
        assert result == "f2_2023_qualifying_round_5.csv"


class TestGetSeriesDirectories:
    @patch("app.core.loader.Path")
    def test_returns_year_directories(self, mock_path):
        mock_series_path = MagicMock()
        mock_series_path.exists.return_value = True

        mock_dir_2023 = MagicMock()
        mock_dir_2023.is_dir.return_value = True
        mock_dir_2023.name = "2023"

        mock_dir_2022 = MagicMock()
        mock_dir_2022.is_dir.return_value = True
        mock_dir_2022.name = "2022"

        mock_file = MagicMock()
        mock_file.is_dir.return_value = False

        mock_series_path.iterdir.return_value = [mock_dir_2023, mock_dir_2022, mock_file]
        mock_path.return_value.__truediv__.return_value = mock_series_path

        result = get_series_directories("F3")
        assert len(result) == 2

    @patch("app.core.loader.Path")
    @patch("app.core.loader.LOGGER")
    def test_nonexistent_series_directory(self, mock_logger, mock_path):
        mock_series_path = MagicMock()
        mock_series_path.exists.return_value = False
        mock_path.return_value.__truediv__.return_value = mock_series_path

        result = get_series_directories("F3")
        assert result == []
        mock_logger.warning.assert_called_once()


class TestLoadAllEntriesData:
    @patch("app.core.loader.get_series_directories")
    @patch("app.core.loader.pd.read_csv")
    def test_loads_multiple_years(self, mock_read_csv, mock_get_dirs):
        mock_dir_2023 = MagicMock()
        mock_dir_2023.name = "2023"
        mock_dir_2022 = MagicMock()
        mock_dir_2022.name = "2022"
        mock_get_dirs.return_value = [mock_dir_2023, mock_dir_2022]

        mock_read_csv.return_value = pd.DataFrame(
            {"Driver": ["Driver1"], "Team": ["Team1"]}
        )

        result = load_all_entries_data("F3")
        assert not result.empty
        assert "year" in result.columns
        assert "series" in result.columns

    @patch("app.core.loader.get_series_directories")
    @patch("app.core.loader.pd.read_csv")
    @patch("app.core.loader.LOGGER")
    def test_handles_missing_file(self, mock_logger, mock_read_csv, mock_get_dirs):
        mock_dir = MagicMock()
        mock_dir.name = "2023"
        mock_get_dirs.return_value = [mock_dir]
        mock_read_csv.side_effect = FileNotFoundError()

        load_all_entries_data("F3")
        mock_logger.warning.assert_called()

    @patch("app.core.loader.get_series_directories")
    def test_empty_result_when_no_data(self, mock_get_dirs):
        mock_get_dirs.return_value = []
        result = load_all_entries_data("F3")
        assert result.empty

    @patch("app.core.loader.get_series_directories")
    @patch("app.core.loader.LOGGER")
    def test_skips_non_integer_directory(self, mock_logger, mock_get_dirs):
        # If year_dir.name is not convertible to int, we should log an error and continue
        bad_dir = MagicMock()
        bad_dir.name = "not-a-year"
        mock_get_dirs.return_value = [bad_dir]

        result = load_all_entries_data("F3")
        assert result.empty
        mock_logger.error.assert_called_once()


class TestLoadYearData:
    @patch("app.core.loader.pd.read_csv")
    def test_loads_driver_data(self, mock_read_csv):
        mock_dir = MagicMock()
        mock_dir.name = "2023"
        mock_read_csv.return_value = pd.DataFrame({"Driver": ["Driver1"], "Pos": [1]})

        result = load_year_data(mock_dir, "F3", "drivers")
        assert result is not None
        assert result["year"].iloc[0] == 2023
        assert result["series"].iloc[0] == "F3"

    @patch("app.core.loader.pd.read_csv")
    def test_drops_na_positions_for_drivers(self, mock_read_csv):
        mock_dir = MagicMock()
        mock_dir.name = "2023"
        mock_read_csv.return_value = pd.DataFrame(
            {"Driver": ["Driver1", "Driver2"], "Pos": [1, None]}
        )

        result = load_year_data(mock_dir, "F3", "drivers")
        assert len(result) == 1

    @patch("app.core.loader.pd.read_csv")
    @patch("app.core.loader.LOGGER")
    def test_handles_file_not_found(self, mock_logger, mock_read_csv):
        mock_dir = MagicMock()
        mock_dir.name = "2023"
        mock_read_csv.side_effect = FileNotFoundError()

        result = load_year_data(mock_dir, "F3", "drivers")
        assert result is None
        mock_logger.warning.assert_called()

    @patch("app.core.loader.LOGGER")
    def test_handles_invalid_year_directory(self, mock_logger):
        # year_dir.name must be non-integer to trigger the ValueError branch
        bad_dir = MagicMock()
        bad_dir.name = "not_a_year"

        result = load_year_data(bad_dir, "F3", "drivers")

        # Should return None because int("not_a_year") raises ValueError
        assert result is None
        mock_logger.error.assert_called_once()


class TestLoadStandingsData:
    @patch("app.core.loader.get_series_directories")
    @patch("app.core.loader.load_year_data")
    def test_concatenates_multiple_years(self, mock_load_year, mock_get_dirs):
        mock_get_dirs.return_value = [MagicMock(), MagicMock()]
        mock_load_year.return_value = pd.DataFrame({"Driver": ["Driver1"]})

        result = load_standings_data("F3", "drivers")
        assert not result.empty

    @patch("app.core.loader.get_series_directories")
    @patch("app.core.loader.load_year_data")
    def test_handles_none_returns(self, mock_load_year, mock_get_dirs):
        mock_get_dirs.return_value = [MagicMock()]
        mock_load_year.return_value = None

        result = load_standings_data("F3", "drivers")
        assert result.empty


class TestLoadQualifyingData:
    @patch("app.core.loader.get_series_directories")
    @patch("app.core.loader.pd.read_csv")
    def test_loads_qualifying_files(self, mock_read_csv, mock_get_dirs):
        mock_dir = MagicMock()
        mock_dir.name = "2023"
        mock_quali_dir = MagicMock()
        mock_quali_dir.exists.return_value = True
        mock_dir.__truediv__.return_value = mock_quali_dir

        mock_file = MagicMock()
        mock_file.stem = "f3_2023_qualifying_round_5"
        mock_quali_dir.glob.return_value = [mock_file]
        mock_get_dirs.return_value = [mock_dir]

        mock_read_csv.return_value = pd.DataFrame({"Driver": ["Driver1"]})

        result = load_qualifying_data("F3")
        assert not result.empty
        assert "round" in result.columns

    @patch("app.core.loader.get_series_directories")
    @patch("app.core.loader.pd.read_csv")
    @patch("app.core.loader.LOGGER")
    def test_handles_parsing_errors(self, mock_logger, mock_read_csv, mock_get_dirs):
        mock_dir = MagicMock()
        mock_dir.name = "2023"
        mock_quali_dir = MagicMock()
        mock_quali_dir.exists.return_value = True
        mock_dir.__truediv__.return_value = mock_quali_dir
        mock_quali_dir.glob.return_value = [MagicMock()]
        mock_get_dirs.return_value = [mock_dir]

        mock_read_csv.side_effect = pd.errors.ParserError()

        load_qualifying_data("F3")
        mock_logger.warning.assert_called()

    @patch("app.core.loader.get_series_directories")
    def test_skips_missing_qualifying_dir(self, mock_get_dirs):
        # If the qualifying directory doesn't exist, function should skip it gracefully
        mock_dir = MagicMock()
        mock_dir.name = "2023"
        mock_quali_dir = MagicMock()
        mock_quali_dir.exists.return_value = False
        mock_dir.__truediv__.return_value = mock_quali_dir
        mock_get_dirs.return_value = [mock_dir]

        result = load_qualifying_data("F3")
        assert result.empty

    @patch("app.core.loader.get_series_directories")
    @patch("app.core.loader.LOGGER")
    def test_logs_error_for_non_integer_year_dir(self, mock_logger, mock_get_dirs):
        # If year_dir.name is not an int, it should be logged and skipped
        mock_dir = MagicMock()
        mock_dir.name = "not_a_year"
        mock_get_dirs.return_value = [mock_dir]

        result = load_qualifying_data("F3")
        assert result.empty
        mock_logger.error.assert_called_once()


class TestGetDriverFilename:
    def test_basic_name(self):
        assert get_driver_filename("Lewis Hamilton") == "lewis_hamilton.json"

    def test_special_characters(self):
        assert get_driver_filename("O'Connor") == "oconnor.json"

    def test_multiple_spaces_and_dashes(self):
        assert get_driver_filename("Jean-Éric Vergne") == "jean_éric_vergne.json"


class TestLoadDriverData:
    @patch("app.core.loader.os.path.exists")
    @patch("builtins.open", new_callable=mock_open)
    def test_loads_profile_data(self, mock_file, mock_exists):
        mock_exists.return_value = True
        mock_file.return_value.read.return_value = json.dumps(
            {"dob": "1990-01-01", "nationality": "British", "scraped": True}
        )

        df = pd.DataFrame({"Driver": ["Lewis Hamilton"]})
        result = load_driver_data(df)

        assert "dob" in result.columns
        assert "nationality" in result.columns

    @patch("app.core.loader.os.path.exists")
    def test_handles_missing_profiles_dir(self, mock_exists):
        mock_exists.return_value = False
        df = pd.DataFrame({"Driver": ["Driver1"]})
        result = load_driver_data(df)

        assert result["dob"].isna().all()

    @patch("app.core.loader.os.path.exists")
    @patch("builtins.open", new_callable=mock_open)
    def test_handles_json_decode_or_other_exception(self, mock_file, mock_exists):
        # Simulate json.load raising an exception so code falls back to default_profile
        mock_exists.return_value = True

        # Make json.load raise; raise when file read attempted
        mock_file.return_value.__enter__.return_value.read.side_effect = ValueError(
            "bad json"
        )

        # to ensure json.load actually runs and raises, patch json.load explicitly
        with patch("json.load", side_effect=ValueError("bad json")):
            df = pd.DataFrame({"Driver": ["Driver1"]})
            result = load_driver_data(df)

        assert "dob" in result.columns
        assert result["dob"].isna().all()


class TestMergeEntries:
    def test_merges_entries_with_drivers(self):
        driver_df = pd.DataFrame(
            {"Driver": ["Driver1"], "year": [2023], "series": ["F3"]}
        )
        entries_df = pd.DataFrame(
            {
                "Driver": ["Driver1"],
                "Team": ["Team1"],
                "Rounds": ["All"],
                "year": [2023],
                "series": ["F3"],
            }
        )

        result = merge_entries(driver_df, entries_df)
        assert "Team" in result.columns
        assert "team_count" in result.columns

    def test_empty_entries(self):
        driver_df = pd.DataFrame({"Driver": ["Driver1"]})
        entries_df = pd.DataFrame()

        result = merge_entries(driver_df, entries_df)
        assert result.equals(driver_df)

    def test_multi_team_driver_selection(self):
        driver_df = pd.DataFrame(
            {"Driver": ["Driver1"], "year": [2023], "series": ["F3"]}
        )
        entries_df = pd.DataFrame(
            {
                "Driver": ["Driver1", "Driver1"],
                "Team": ["Team1", "Team2"],
                "Rounds": ["1-5", "6-12"],
                "year": [2023, 2023],
                "series": ["F3", "F3"],
            }
        )

        result = merge_entries(driver_df, entries_df)
        assert result["Team"].iloc[0] == "Team2"


class TestParseRoundCount:
    def test_all_rounds(self):
        assert parse_round_count("All") == float("inf")

    def test_single_round(self):
        assert parse_round_count("5") == 1

    def test_range(self):
        assert parse_round_count("1-5") == 5

    def test_multiple_ranges(self):
        assert parse_round_count("1-3,5,7-9") == 7

    def test_with_endash(self):
        assert parse_round_count("1–5") == 5

    def test_none_value(self):
        assert parse_round_count(None) == float("inf")


class TestMergeTeamData:
    def test_merges_team_standings(self):
        driver_df = pd.DataFrame(
            {"Driver": ["Driver1"], "Team": ["Team1"], "year": [2023]}
        )
        team_df = pd.DataFrame(
            {"Team": ["Team1"], "Pos": [1], "Points": [100], "year": [2023]}
        )

        result = merge_team_data(driver_df, team_df)
        assert "team_pos" in result.columns
        assert "team_points" in result.columns
        assert result["team_pos"].iloc[0] == 1


class TestLoadData:
    @patch("app.core.loader.load_standings_data")
    @patch("app.core.loader.load_all_entries_data")
    @patch("app.core.loader.merge_entries")
    @patch("app.core.loader.merge_team_data")
    @patch("app.core.loader.load_driver_data")
    def test_full_pipeline(
        self,
        mock_load_driver,
        mock_merge_team,
        mock_merge_entries,
        mock_load_entries,
        mock_load_standings,
    ):
        mock_df = pd.DataFrame({"Driver": ["Driver1"]})
        mock_load_standings.return_value = mock_df
        mock_load_entries.return_value = mock_df
        mock_merge_entries.return_value = mock_df
        mock_merge_team.return_value = mock_df
        mock_load_driver.return_value = mock_df

        result = load_data("F3")
        assert not result.empty
        mock_load_standings.assert_called()
        mock_load_driver.assert_called_once()
