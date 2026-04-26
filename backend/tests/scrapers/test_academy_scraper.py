import os
from unittest.mock import MagicMock, mock_open, patch

import pytest

from app.config import DATA_DIR
from app.scrapers.academy_scraper import (
    expand_years_in_data,
    extract_table_data,
    extract_team_links,
    parse_year_range,
    save_academy_data,
    scrape_academies,
    scrape_academy_page,
)


class TestExtractTeamLinks:
    @pytest.fixture
    def mock_soup(self):
        """Create a mock BeautifulSoup object with F1 section and table."""
        soup = MagicMock()

        # Create mock F1 heading
        f1_heading = MagicMock()
        f1_heading.find_next.return_value = MagicMock()

        # Create mock table with rows
        table = MagicMock()
        row1 = MagicMock()  # Header row
        row2 = MagicMock()  # Data row with link
        row3 = MagicMock()  # Data row without link

        # Set up cells for row2
        cell1 = MagicMock()
        cell2 = MagicMock()
        link = MagicMock()
        link.get.return_value = "/wiki/Red_Bull_Junior_Team"
        link.text = "Red Bull Junior Team"
        cell1.find.return_value = link

        row2.find_all.return_value = [cell1, cell2]

        # Set up cells for row3 (no link)
        cell3 = MagicMock()
        cell3.find.return_value = None
        cell4 = MagicMock()
        row3.find_all.return_value = [cell3, cell4]

        table.find_all.return_value = [row1, row2, row3]

        # Set up soup mock
        soup.find.return_value = f1_heading
        f1_heading.find_next.return_value = table

        return soup

    def test_no_f1_section(self, mock_soup):
        """Test handling when no F1 section is found."""
        mock_soup.find.return_value = None
        result = extract_team_links(mock_soup)
        assert result == []

    def test_no_table_found(self, mock_soup):
        """Test handling when no table is found."""
        f1_heading = mock_soup.find.return_value
        f1_heading.find_next.return_value = None
        result = extract_team_links(mock_soup)
        assert result == []

    def test_invalid_link_format(self, mock_soup):
        """Test handling of invalid link formats."""
        table = mock_soup.find.return_value.find_next.return_value
        row2 = table.find_all.return_value[1]
        cell1 = row2.find_all.return_value[0]
        link = cell1.find.return_value
        link.get.return_value = "/invalid/path"
        result = extract_team_links(mock_soup)
        assert result == []


class TestParseYearRange:
    def test_single_year(self):
        """Test parsing a single year."""
        assert parse_year_range("2023") == ["2023"]

    def test_year_range(self):
        """Test parsing a year range."""
        assert parse_year_range("2010–2015") == [
            "2010",
            "2011",
            "2012",
            "2013",
            "2014",
            "2015",
        ]

    def test_multiple_ranges(self):
        """Test parsing multiple year ranges."""
        assert parse_year_range("2010–2012, 2015–2017") == [
            "2010",
            "2011",
            "2012",
            "2015",
            "2016",
            "2017",
        ]

    def test_open_ended_range(self):
        """Test parsing open-ended range (should use CURRENT_YEAR)."""
        with patch("app.scrapers.academy_scraper.CURRENT_YEAR", 2024):
            assert parse_year_range("2020–") == ["2020", "2021", "2022", "2023", "2024"]

    def test_invalid_year_format(self):
        """Test handling invalid year formats."""
        assert parse_year_range("unknown") == ["unknown"]
        assert parse_year_range("2010–invalid") == ["2010–invalid"]


class TestExpandYearsInData:
    def test_empty_data(self):
        """Test handling empty data."""
        headers, data = expand_years_in_data([], [])
        assert headers == ["Driver", "Year"]
        assert data == []

    def test_no_year_column(self):
        """Test handling when no year column is found."""
        headers = ["Driver", "Team"]
        data_rows = [["Lewis Hamilton", "Mercedes"]]
        headers, data = expand_years_in_data(headers, data_rows)
        assert headers == ["Driver", "Year"]
        assert data == []

    def test_single_year_expansion(self):
        """Test expanding single year data."""
        headers = ["Driver", "Years"]
        data_rows = [["Lewis Hamilton", "2023"]]
        headers, data = expand_years_in_data(headers, data_rows)
        assert headers == ["Driver", "Year"]
        assert data == [["Lewis Hamilton", "2023"]]

    def test_year_range_expansion(self):
        """Test expanding year ranges."""
        headers = ["Driver", "Years"]
        data_rows = [["Lewis Hamilton", "2020–2022"]]
        headers, data = expand_years_in_data(headers, data_rows)
        assert headers == ["Driver", "Year"]
        assert data == [
            ["Lewis Hamilton", "2020"],
            ["Lewis Hamilton", "2021"],
            ["Lewis Hamilton", "2022"],
        ]


class TestExtractTableData:
    @pytest.fixture
    def mock_table(self):
        """Create a mock table with headers and data."""
        table = MagicMock()

        # Create header row
        header_row = MagicMock()
        th1 = MagicMock()
        th1.get.return_value = "1"
        th2 = MagicMock()
        header_row.find_all.return_value = [th1, th2]

        # Create data row
        data_row = MagicMock()
        td1 = MagicMock()
        td2 = MagicMock()
        data_row.find_all.return_value = [td1, td2]

        table.find_all.return_value = [header_row, data_row]

        return table

    def test_empty_table(self):
        """Test handling empty table."""
        table = MagicMock()
        table.find_all.return_value = []
        result = extract_table_data(table, "current_drivers")
        assert result is None

    @patch("app.scrapers.academy_scraper.remove_superscripts")
    def test_f1_graduates_table(self, mock_remove_superscripts, mock_table):
        """Test extraction from F1 graduates table with multiple header rows."""
        mock_remove_superscripts.side_effect = (
            lambda x, *_: x.text if hasattr(x, "text") else x
        )

        # Create second header row for F1 graduates
        second_header_row = MagicMock()
        th3 = MagicMock()
        th4 = MagicMock()
        second_header_row.find_all.return_value = [th3, th4]

        # Modify table to have 3 rows (header, second header, data)
        mock_table.find_all.return_value = [
            mock_table.find_all.return_value[0],  # First header row
            second_header_row,  # Second header row
            mock_table.find_all.return_value[1],  # Data row
        ]

        result = extract_table_data(mock_table, "f1_graduates")

        assert result is not None
        assert len(result["headers"]) > 0


class TestScrapeAcademyPage:
    @pytest.fixture
    def mock_session(self):
        """Create a mock session."""
        session = MagicMock()
        response = MagicMock()
        response.text = "<html><body></body></html>"
        response.close = MagicMock()
        session.get.return_value = response
        return session

    @patch("app.scrapers.academy_scraper.safe_request")
    def test_failed_request(self, mock_safe_request, mock_session):
        """Test handling failed request."""
        mock_safe_request.return_value = None
        result = scrape_academy_page(
            "https://example.com",
            "Red Bull Junior Team",
            mock_session,
        )
        assert result is None


class TestSaveAcademyData:
    @patch("builtins.open", new_callable=mock_open)
    @patch("os.makedirs")
    def test_save_academy_data_success(self, mock_makedirs, mock_file):
        """Test successful saving of academy data."""
        academy_data = {
            "name": "Red Bull Junior Team",
            "current_drivers": [
                {"headers": ["Driver", "Year"], "data": [["Lewis Hamilton", "2023"]]},
            ],
            "former_drivers": [
                {"headers": ["Driver", "Year"], "data": [["Sebastian Vettel", "2008"]]},
            ],
            "f1_graduates": [
                {"headers": ["Driver", "Year"], "data": [["Max Verstappen", "2015"]]},
            ],
        }

        academies_dir = os.path.join(DATA_DIR, "academies")
        save_academy_data(academy_data, academies_dir)

        # Verify makedirs was called
        mock_makedirs.assert_called_once_with(academies_dir, exist_ok=True)

        # Verify file was opened correctly
        mock_file.assert_called_once_with(
            os.path.join(academies_dir, "Red_Bull_Junior_Team_drivers.csv"),
            "w",
            newline="",
            encoding="utf-8",
        )

        # Verify CSV content was written
        handle = mock_file()
        handle.write.assert_any_call("Driver,Year\r\n")
        handle.write.assert_any_call("Lewis Hamilton,2023\r\n")
        handle.write.assert_any_call("Sebastian Vettel,2008\r\n")
        handle.write.assert_any_call("Max Verstappen,2015\r\n")

    @patch("os.makedirs")
    def test_empty_academy_data(self, mock_makedirs):
        """Test handling empty academy data."""
        save_academy_data(None, "test_dir")
        mock_makedirs.assert_not_called()


class TestScrapeAcademies:
    @patch("app.scrapers.academy_scraper.create_session")
    @patch("app.scrapers.academy_scraper.safe_request")
    @patch("app.scrapers.academy_scraper.BeautifulSoup")
    @patch("app.scrapers.academy_scraper.extract_team_links")
    @patch("app.scrapers.academy_scraper.scrape_academy_page")
    @patch("app.scrapers.academy_scraper.save_academy_data")
    def test_full_scrape_workflow(
        self,
        mock_save,
        mock_scrape_page,
        mock_extract_links,
        mock_beautifulsoup,
        mock_safe_request,
        mock_create_session,
    ):
        """Test the full scrape workflow."""
        # Setup mock session
        mock_session = MagicMock()
        mock_create_session.return_value = mock_session

        # Setup mock response
        mock_response = MagicMock()
        mock_response.text = "<html><body></body></html>"
        mock_response.close = MagicMock()
        mock_safe_request.return_value = mock_response

        # Setup mock soup
        mock_soup = MagicMock()
        mock_beautifulsoup.return_value = mock_soup

        # Setup team links
        mock_extract_links.return_value = [
            {"name": "Red Bull Junior Team", "url": "https://example.com/red-bull"},
            {
                "name": "Marussia F1 Team Young Driver Program",
                "url": "https://example.com/marussia",
            },  # This should be skipped
        ]

        # Setup academy page data
        mock_scrape_page.side_effect = [
            {
                "name": "Red Bull Junior Team",
                "current_drivers": [],
                "former_drivers": [],
                "f1_graduates": [],
            },
            None,  # Marussia should be skipped
        ]

        scrape_academies()

        # Verify session was created
        mock_create_session.assert_called_once()

        # Verify main page was requested
        mock_safe_request.assert_called_once_with(
            mock_session,
            "https://en.wikipedia.org/wiki/Driver_development_program",
        )

        # Verify team links were extracted
        mock_extract_links.assert_called_once_with(mock_soup)

        # Verify only one academy was scraped (Marussia was skipped)
        assert mock_scrape_page.call_count == 1
        mock_scrape_page.assert_any_call(
            "https://example.com/red-bull",
            "Red Bull Junior Team",
            mock_session,
        )

        # Verify data was saved
        mock_save.assert_called_once()

    @patch("app.scrapers.academy_scraper.create_session")
    @patch("app.scrapers.academy_scraper.safe_request")
    def test_failed_initial_request(self, mock_safe_request, mock_create_session):
        """Test handling failed initial request."""
        mock_session = MagicMock()
        mock_create_session.return_value = mock_session
        mock_safe_request.return_value = None

        result = scrape_academies()
        assert result is None
        mock_safe_request.assert_called_once()
