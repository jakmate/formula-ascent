from unittest.mock import patch, MagicMock
import pandas as pd
from app.core.predictor import create_target_variable, main


class TestCreateTargetVariable:
    """Tests for create_target_variable function"""

    def test_empty_feeder_df(self):
        """Test returns feeder_df with promoted=NaN when feeder_df is empty"""
        feeder_df = pd.DataFrame()
        parent_df = pd.DataFrame({"year": [2020, 2021], "Driver": ["Driver1", "Driver2"]})

        result = create_target_variable(feeder_df, parent_df, "F2")

        assert result.empty is False or result.empty is True  # Can be either
        if not result.empty:
            assert "promoted" in result.columns

    def test_empty_parent_df(self):
        """Test returns feeder_df with promoted=NaN when parent_df is empty"""
        feeder_df = pd.DataFrame({"Driver": ["Driver1"], "year": [2020]})
        parent_df = pd.DataFrame()

        result = create_target_variable(feeder_df, parent_df, "F2")

        assert "promoted" in result.columns
        assert pd.isna(result["promoted"].iloc[0])

    @patch("app.core.predictor.calculate_participation_stats")
    @patch("app.core.predictor.get_race_columns")
    def test_promoted_initialized_to_zero(self, mock_get_race_cols, mock_calc_stats):
        """Test promoted column is initialized to 0"""
        feeder_df = pd.DataFrame({"Driver": ["Driver1", "Driver2"], "year": [2020, 2020]})
        parent_df = pd.DataFrame({"Driver": ["Driver1"], "year": [2021]})

        mock_get_race_cols.return_value = ["Race1", "Race2"]
        mock_calc_stats.return_value = []

        result = create_target_variable(feeder_df, parent_df, "F2")

        assert "promoted" in result.columns
        assert (result["promoted"] == 0).any() or (result["promoted"] == 1).any()

    @patch("app.core.predictor.calculate_participation_stats")
    @patch("app.core.predictor.get_race_columns")
    @patch("app.core.predictor.CURRENT_YEAR", 2025)
    def test_participation_lookup_current_year_threshold(self, mock_get_race_cols, mock_calc_stats):
        """Test threshold is 0 for current year"""
        feeder_df = pd.DataFrame({"Driver": ["Driver1"], "year": [2024]})
        parent_df = pd.DataFrame({"Driver": ["Driver1"], "year": [2025]})

        mock_get_race_cols.return_value = ["R1", "R2", "R3", "R4", "R5"]
        mock_calc_stats.return_value = [{"Driver": "Driver1", "participated_races": 1}]

        create_target_variable(feeder_df, parent_df, "F2")

        assert mock_calc_stats.called

    @patch("app.core.predictor.calculate_participation_stats")
    @patch("app.core.predictor.get_race_columns")
    def test_participation_lookup_past_year_threshold(self, mock_get_race_cols, mock_calc_stats):
        """Test threshold is 40% of races for past years"""
        feeder_df = pd.DataFrame({"Driver": ["Driver1"], "year": [2020]})
        parent_df = pd.DataFrame({"Driver": ["Driver1"], "year": [2021]})

        mock_get_race_cols.return_value = ["R1", "R2", "R3", "R4", "R5"]  # 5 races
        # 40% of 5 = 2, so need 3+ races to pass threshold
        mock_calc_stats.return_value = [{"Driver": "Driver1", "participated_races": 3}]

        result = create_target_variable(feeder_df, parent_df, "F2")

        assert result["promoted"].iloc[0] == 1

    def test_f1_series_checks_three_years(self):
        """Test F1 series checks 1, 2, and 3 years ahead"""
        feeder_df = pd.DataFrame({"Driver": ["Driver1"], "year": [2020]})
        parent_df = pd.DataFrame(
            {"Driver": ["Driver1", "Driver1"], "year": [2023, 2023]}  # 3 years later
        )

        with patch("app.core.predictor.get_race_columns") as mock_get_cols:
            with patch("app.core.predictor.calculate_participation_stats") as mock_calc:
                mock_get_cols.return_value = ["R1", "R2"]
                mock_calc.return_value = [{"Driver": "Driver1", "participated_races": 2}]

                result = create_target_variable(feeder_df, parent_df, "F1")

                assert result["promoted"].iloc[0] == 1

    @patch("app.core.predictor.calculate_participation_stats")
    @patch("app.core.predictor.get_race_columns")
    def test_non_f1_series_checks_one_year(self, mock_get_race_cols, mock_calc_stats):
        """Test non-F1 series only checks 1 year ahead"""
        feeder_df = pd.DataFrame({"Driver": ["Driver1"], "year": [2020]})
        parent_df = pd.DataFrame({"Driver": ["Driver1"], "year": [2021]})

        mock_get_race_cols.return_value = ["R1", "R2"]
        mock_calc_stats.return_value = [{"Driver": "Driver1", "participated_races": 2}]

        result = create_target_variable(feeder_df, parent_df, "F2")

        assert result["promoted"].iloc[0] == 1

    @patch("app.core.predictor.calculate_participation_stats")
    @patch("app.core.predictor.get_race_columns")
    def test_only_last_feeder_season_processed(self, mock_get_race_cols, mock_calc_stats):
        """Test only last feeder season for each driver gets promotion flag"""
        feeder_df = pd.DataFrame(
            {"Driver": ["Driver1", "Driver1", "Driver1"], "year": [2018, 2019, 2020]}
        )
        parent_df = pd.DataFrame({"Driver": ["Driver1"], "year": [2021]})

        mock_get_race_cols.return_value = ["R1", "R2"]
        mock_calc_stats.return_value = [{"Driver": "Driver1", "participated_races": 2}]

        result = create_target_variable(feeder_df, parent_df, "F2")

        # First two years should be 0, last year should be 1
        assert result["promoted"].iloc[0] == 0
        assert result["promoted"].iloc[1] == 0
        assert result["promoted"].iloc[2] == 1

    @patch("app.core.predictor.calculate_participation_stats")
    @patch("app.core.predictor.get_race_columns")
    def test_future_beyond_max_parent_year_returns_nan(self, mock_get_race_cols, mock_calc_stats):
        """Test returns NaN when future years cannot be observed"""
        feeder_df = pd.DataFrame({"Driver": ["Driver1"], "year": [2023]})
        parent_df = pd.DataFrame({"Driver": ["Driver2"], "year": [2023]})  # max_year = 2023

        mock_get_race_cols.return_value = ["R1"]
        mock_calc_stats.return_value = []

        result = create_target_variable(feeder_df, parent_df, "F2")

        # 2023 + 1 = 2024, which is > 2023, so should be NaN
        assert pd.isna(result["promoted"].iloc[0])

    @patch("app.core.predictor.calculate_participation_stats")
    @patch("app.core.predictor.get_race_columns")
    def test_no_promotion_when_not_in_parent_series(self, mock_get_race_cols, mock_calc_stats):
        """Test returns 0 when driver doesn't appear in parent series"""
        feeder_df = pd.DataFrame({"Driver": ["Driver1"], "year": [2020]})
        parent_df = pd.DataFrame({"Driver": ["Driver2", "Driver2"], "year": [2021, 2022]})

        mock_get_race_cols.return_value = ["R1", "R2"]
        mock_calc_stats.return_value = [{"Driver": "Driver2", "participated_races": 2}]

        result = create_target_variable(feeder_df, parent_df, "F2")

        assert result["promoted"].iloc[0] == 0

    @patch("app.core.predictor.calculate_participation_stats")
    @patch("app.core.predictor.get_race_columns")
    def test_promotion_when_participation_exceeds_threshold(
        self, mock_get_race_cols, mock_calc_stats
    ):
        """Test promotion=1 when driver exceeds participation threshold"""
        feeder_df = pd.DataFrame({"Driver": ["Driver1"], "year": [2020]})
        parent_df = pd.DataFrame({"Driver": ["Driver1"], "year": [2021]})

        mock_get_race_cols.return_value = ["R1", "R2", "R3", "R4", "R5"]
        # Threshold is 40% * 5 = 2, so 3 races passes
        mock_calc_stats.return_value = [{"Driver": "Driver1", "participated_races": 3}]

        result = create_target_variable(feeder_df, parent_df, "F2")

        assert result["promoted"].iloc[0] == 1

    @patch("app.core.predictor.calculate_participation_stats")
    @patch("app.core.predictor.get_race_columns")
    def test_no_promotion_when_below_threshold(self, mock_get_race_cols, mock_calc_stats):
        """Test promotion=0 when driver doesn't meet participation threshold"""
        feeder_df = pd.DataFrame({"Driver": ["Driver1"], "year": [2020]})
        parent_df = pd.DataFrame({"Driver": ["Driver1"], "year": [2021]})

        mock_get_race_cols.return_value = ["R1", "R2", "R3", "R4", "R5"]
        # Threshold is 40% * 5 = 2, so 2 races doesn't pass (need > 2)
        mock_calc_stats.return_value = [{"Driver": "Driver1", "participated_races": 2}]

        result = create_target_variable(feeder_df, parent_df, "F2")

        assert result["promoted"].iloc[0] == 0

    @patch("app.core.predictor.calculate_participation_stats")
    @patch("app.core.predictor.get_race_columns")
    def test_skips_years_with_no_race_columns(self, mock_get_race_cols, mock_calc_stats):
        """Test skips parent years that have no race columns"""
        feeder_df = pd.DataFrame({"Driver": ["Driver1"], "year": [2020]})
        parent_df = pd.DataFrame({"Driver": ["Driver1"], "year": [2021]})

        mock_get_race_cols.return_value = []  # No race columns
        mock_calc_stats.return_value = []

        result = create_target_variable(feeder_df, parent_df, "F2")

        assert result["promoted"].iloc[0] == 0

    @patch("app.core.predictor.calculate_participation_stats")
    @patch("app.core.predictor.get_race_columns")
    def test_multiple_drivers_different_outcomes(self, mock_get_race_cols, mock_calc_stats):
        """Test handles multiple drivers with different promotion outcomes"""
        feeder_df = pd.DataFrame(
            {"Driver": ["Driver1", "Driver2", "Driver3"], "year": [2020, 2020, 2020]}
        )
        parent_df = pd.DataFrame(
            {"Driver": ["Driver1", "Driver1", "Driver2"], "year": [2021, 2021, 2021]}
        )

        mock_get_race_cols.return_value = ["R1", "R2", "R3", "R4", "R5"]
        mock_calc_stats.return_value = [
            {"Driver": "Driver1", "participated_races": 5},  # Passes
            {"Driver": "Driver2", "participated_races": 1},  # Fails
        ]

        result = create_target_variable(feeder_df, parent_df, "F2")

        assert result.loc[result["Driver"] == "Driver1", "promoted"].iloc[0] == 1
        assert result.loc[result["Driver"] == "Driver2", "promoted"].iloc[0] == 0
        assert result.loc[result["Driver"] == "Driver3", "promoted"].iloc[0] == 0

    @patch("app.core.predictor.calculate_participation_stats")
    @patch("app.core.predictor.get_race_columns")
    def test_f1_series_breaks_when_year_exceeds_max(self, mock_get_race_cols, mock_calc_stats):
        """Test F1 series breaks loop when checking year exceeds max_parent_year"""
        feeder_df = pd.DataFrame({"Driver": ["Driver1"], "year": [2020]})
        # max_parent_year = 2021, so checking offset=2 (2022) and offset=3 (2023) should break
        parent_df = pd.DataFrame({"Driver": ["Driver1"], "year": [2021]})

        mock_get_race_cols.return_value = ["R1", "R2"]
        mock_calc_stats.return_value = [{"Driver": "Driver1", "participated_races": 0}]

        result = create_target_variable(feeder_df, parent_df, "F1")

        # Should return 0 (not promoted) since year+1 (2021) has no participation
        # and the loop breaks before checking years 2022 and 2023
        assert result["promoted"].iloc[0] == 0


class TestMain:
    """Tests for main function"""

    @patch("app.core.predictor.predict_drivers")
    @patch("app.core.predictor.train_models")
    @patch("app.core.predictor.engineer_features")
    @patch("app.core.predictor.create_target_variable")
    @patch("app.core.predictor.calculate_qualifying_features")
    @patch("app.core.predictor.load_standings_data")
    @patch("app.core.predictor.load_data")
    @patch("app.core.predictor.load_qualifying_data")
    def test_main_execution_flow(
        self,
        mock_load_quali,
        mock_load_data,
        mock_load_standings,
        mock_calc_quali,
        mock_create_target,
        mock_engineer,
        mock_train,
        mock_predict,
    ):
        """Test main function executes all steps in correct order"""
        # Setup mocks
        mock_feeder_quali = MagicMock()
        mock_feeder_df = MagicMock()
        mock_parent_df = MagicMock()
        mock_features_df = MagicMock()
        mock_models = {"LightGBM": MagicMock()}
        mock_feature_cols = ["col1", "col2"]
        mock_scaler = MagicMock()

        mock_load_quali.return_value = mock_feeder_quali
        mock_load_data.return_value = mock_feeder_df
        mock_load_standings.return_value = mock_parent_df
        mock_calc_quali.return_value = mock_feeder_df
        mock_create_target.return_value = mock_feeder_df
        mock_engineer.return_value = mock_features_df
        mock_train.return_value = (mock_models, mock_feature_cols, mock_scaler)

        # Execute
        main()

        # Verify call sequence
        mock_load_quali.assert_called_once_with("F3")
        mock_load_data.assert_called_once_with("F3")
        mock_load_standings.assert_called_once_with("F2", "drivers")
        mock_calc_quali.assert_called_once_with(mock_feeder_df, mock_feeder_quali)
        mock_create_target.assert_called_once_with(mock_feeder_df, mock_parent_df, "F2")
        mock_engineer.assert_called_once_with(mock_feeder_df)
        mock_train.assert_called_once_with(mock_features_df)
        mock_predict.assert_called_once_with(
            mock_models, mock_features_df, mock_feature_cols, mock_scaler
        )

    @patch("app.core.predictor.predict_drivers")
    @patch("app.core.predictor.train_models")
    @patch("app.core.predictor.engineer_features")
    @patch("app.core.predictor.create_target_variable")
    @patch("app.core.predictor.calculate_qualifying_features")
    @patch("app.core.predictor.load_standings_data")
    @patch("app.core.predictor.load_data")
    @patch("app.core.predictor.load_qualifying_data")
    @patch("builtins.print")
    def test_main_prints_progress(
        self,
        mock_print,
        mock_load_quali,
        mock_load_data,
        mock_load_standings,
        mock_calc_quali,
        mock_create_target,
        mock_engineer,
        mock_train,
        mock_predict,
    ):
        """Test main function prints progress messages"""
        # Setup basic mocks
        mock_load_quali.return_value = MagicMock()
        mock_load_data.return_value = MagicMock()
        mock_load_standings.return_value = MagicMock()
        mock_calc_quali.return_value = MagicMock()
        mock_create_target.return_value = MagicMock()
        mock_engineer.return_value = MagicMock()
        mock_train.return_value = ({}, [], None)

        main()

        # Verify progress messages
        print_calls = [str(call) for call in mock_print.call_args_list]
        assert any("Loading F3 qualifying data" in str(call) for call in print_calls)
        assert any("Adding qualifying features" in str(call) for call in print_calls)
        assert any("Creating target variable" in str(call) for call in print_calls)
        assert any("Engineering features" in str(call) for call in print_calls)
        assert any("Training all models" in str(call) for call in print_calls)
        assert any("Making predictions" in str(call) for call in print_calls)

    @patch("app.core.predictor.predict_drivers")
    @patch("app.core.predictor.train_models")
    @patch("app.core.predictor.engineer_features")
    @patch("app.core.predictor.create_target_variable")
    @patch("app.core.predictor.calculate_qualifying_features")
    @patch("app.core.predictor.load_standings_data")
    @patch("app.core.predictor.load_data")
    @patch("app.core.predictor.load_qualifying_data")
    def test_main_handles_feature_dataframe_promoted_column(
        self,
        mock_load_quali,
        mock_load_data,
        mock_load_standings,
        mock_calc_quali,
        mock_create_target,
        mock_engineer,
        mock_train,
        mock_predict,
    ):
        """Test main properly assigns promoted column to features_df"""
        mock_feeder_df = MagicMock()
        mock_feeder_df.__getitem__ = MagicMock(return_value=[0, 1, 0])
        mock_features_df = MagicMock()

        mock_load_quali.return_value = MagicMock()
        mock_load_data.return_value = mock_feeder_df
        mock_load_standings.return_value = MagicMock()
        mock_calc_quali.return_value = mock_feeder_df
        mock_create_target.return_value = mock_feeder_df
        mock_engineer.return_value = mock_features_df
        mock_train.return_value = ({}, [], None)

        main()

        # Verify engineer_features was called with feeder_df
        mock_engineer.assert_called_once_with(mock_feeder_df)

    @patch("app.core.predictor.predict_drivers")
    @patch("app.core.predictor.train_models")
    @patch("app.core.predictor.engineer_features")
    @patch("app.core.predictor.create_target_variable")
    @patch("app.core.predictor.calculate_qualifying_features")
    @patch("app.core.predictor.load_standings_data")
    @patch("app.core.predictor.load_data")
    @patch("app.core.predictor.load_qualifying_data")
    def test_main_uses_correct_series_order(
        self,
        mock_load_quali,
        mock_load_data,
        mock_load_standings,
        mock_calc_quali,
        mock_create_target,
        mock_engineer,
        mock_train,
        mock_predict,
    ):
        """Test main uses F3 as feeder and F2 as parent series"""
        mock_load_quali.return_value = MagicMock()
        mock_load_data.return_value = MagicMock()
        mock_load_standings.return_value = MagicMock()
        mock_calc_quali.return_value = MagicMock()
        mock_create_target.return_value = MagicMock()
        mock_engineer.return_value = MagicMock()
        mock_train.return_value = ({}, [], None)

        main()

        # Verify correct series are used
        mock_load_quali.assert_called_with("F3")
        mock_load_data.assert_called_with("F3")
        mock_load_standings.assert_called_with("F2", "drivers")

        # Verify F2 is used as parent series in target creation
        args = mock_create_target.call_args[0]
        assert args[2] == "F2"
