import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch
from services.data_extractor_service import DataExtractorService


class TestDataExtractorService:
    """
    DataExtractorServiceのユニットテスト
    """

    @patch("os.listdir")
    @patch("pandas.read_excel")
    def test_extract_from_directory_success(self, mock_read_excel, mock_listdir):
        """正常なケースでデータが正しく抽出されるかテスト"""
        # Mock setup
        mock_listdir.return_value = ["test_data.xlsx", "other_file.txt"]
        mock_df = pd.DataFrame(
            {
                "time": [1950, 1951, 1952],
                "value": [100, 110, 120],
            }
        )
        mock_read_excel.return_value = mock_df

        # Execution
        time_data, value_data, filename = DataExtractorService.extract_from_directory(
            "dummy_dir"
        )

        # Assertion
        np.testing.assert_array_equal(time_data, mock_df["time"].values)
        np.testing.assert_array_equal(value_data, mock_df["value"].values)
        assert filename == "test_data.xlsx"
        mock_read_excel.assert_called_once_with(
            "dummy_dir/test_data.xlsx", header=0, engine="openpyxl"
        )

    @patch("os.listdir")
    def test_extract_no_xlsx_file_found(self, mock_listdir):
        """xlsxファイルが見つからない場合にFileNotFoundErrorを送出するかテスト"""
        mock_listdir.return_value = ["file1.txt", "file2.csv"]
        with pytest.raises(FileNotFoundError):
            DataExtractorService.extract_from_directory("dummy_dir")

    @patch("os.listdir")
    @patch("pandas.read_excel")
    def test_extract_invalid_columns(self, mock_read_excel, mock_listdir):
        """Excelの列名が不正な場合にValueErrorを送出するかテスト"""
        mock_listdir.return_value = ["test_data.xlsx"]
        mock_df = pd.DataFrame({"year": [1950], "population": [100]})
        mock_read_excel.return_value = mock_df

        with pytest.raises(ValueError, match="列名に 'time', 'value' が必要です"):
            DataExtractorService.extract_from_directory("dummy_dir")

    @patch("os.listdir")
    @patch("pandas.read_excel")
    def test_extract_non_numeric_data(self, mock_read_excel, mock_listdir):
        """データが非数値の場合にValueErrorを送出するかテスト"""
        mock_listdir.return_value = ["test_data.xlsx"]
        mock_df = pd.DataFrame({"time": [1950], "value": ["invalid"]})
        mock_read_excel.return_value = mock_df

        with pytest.raises(
            ValueError, match="'time'列と'value'列は数値データである必要があります"
        ):
            DataExtractorService.extract_from_directory("dummy_dir")

    @patch("os.listdir")
    @patch("pandas.read_excel")
    def test_extract_data_with_nan(self, mock_read_excel, mock_listdir):
        """データにNaNが含まれる場合にValueErrorを送出するかテスト"""
        mock_listdir.return_value = ["test_data.xlsx"]
        mock_df = pd.DataFrame({"time": [1950, 1951], "value": [100, np.nan]})
        mock_read_excel.return_value = mock_df

        with pytest.raises(ValueError, match="データにNaN値が含まれています"):
            DataExtractorService.extract_from_directory("dummy_dir")
