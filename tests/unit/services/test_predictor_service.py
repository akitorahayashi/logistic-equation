import pytest
import numpy as np
from unittest.mock import MagicMock, patch
from services.predictor_service import FuturePredictorService
from models.logistic_equation import LogisticEquationModel
from schemas.prediction_settings_schema import PredictionSettingsSchema


@pytest.fixture
def mock_equation():
    """LogisticEquationModelのモック"""
    mock = MagicMock(spec=LogisticEquationModel)
    mock.gamma = 0.1
    mock.K = 1000
    mock.solve_runge_kutta.return_value = (
        np.array([10, 20, 30]),
        np.array([100, 110, 120]),
    )
    return mock


@pytest.fixture
def mock_settings():
    """PredictionSettingsSchemaのモック"""
    mock = MagicMock(spec=PredictionSettingsSchema)
    mock.forecast_end_t = 100
    return mock


class TestFuturePredictorService:
    """
    FuturePredictorServiceのユニットテスト
    """

    def test_init(self, mock_equation, mock_settings):
        """初期化をテスト"""
        predictor = FuturePredictorService(mock_equation, mock_settings)
        assert predictor.equation == mock_equation
        assert predictor.prediction_settings == mock_settings

    def test_predict(self, mock_equation, mock_settings):
        """predictメソッドが正しくsolve_runge_kuttaを呼び出すかテスト"""
        time_array = np.array([0, 1, 2])
        value_array = np.array([50, 55, 60])
        predictor = FuturePredictorService(mock_equation, mock_settings)

        t_forecast, v_forecast = predictor.predict(time_array, value_array)

        # solve_runge_kuttaが正しい引数で呼ばれたか確認
        mock_equation.solve_runge_kutta.assert_called_once_with(
            50, 0, mock_settings.forecast_end_t, 0.1
        )
        assert t_forecast is not None
        assert v_forecast is not None

    def test_predict_invalid_time_raises_error(self, mock_equation, mock_settings):
        """予測終了時刻が開始時刻以前の場合にValueErrorを送出するかテスト"""
        mock_settings.forecast_end_t = -10  # 開始時刻(0)より前
        time_array = np.array([0, 1])
        value_array = np.array([50, 55])
        predictor = FuturePredictorService(mock_equation, mock_settings)
        with pytest.raises(ValueError):
            predictor.predict(time_array, value_array)

    @patch("os.makedirs")
    @patch("pandas.DataFrame.to_excel")
    def test_save_prediction_to_excel(
        self, mock_to_excel, mock_makedirs, mock_equation, mock_settings
    ):
        """予測結果が正しくExcelに保存されるか（の呼び出しが行われるか）テスト"""
        predictor = FuturePredictorService(mock_equation, mock_settings)
        t_forecast = np.array([0, 1, 2])
        v_forecast = np.array([10, 20, 30])

        with patch("config.config.OUTPUT_DIR", "mock_output"):
            path = predictor.save_prediction_to_excel(
                t_forecast, v_forecast, "test_output", interval=1
            )

        mock_makedirs.assert_called_once_with("mock_output", exist_ok=True)
        mock_to_excel.assert_called_once()
        assert path == "mock_output/test_output.xlsx"

    def test_save_prediction_to_excel_invalid_interval(
        self, mock_equation, mock_settings
    ):
        """不正なintervalでValueErrorを送出するかテスト"""
        predictor = FuturePredictorService(mock_equation, mock_settings)
        with pytest.raises(ValueError):
            predictor.save_prediction_to_excel(
                np.array([]), np.array([]), "test", interval=0
            )
        with pytest.raises(ValueError):
            predictor.save_prediction_to_excel(
                np.array([]), np.array([]), "test", interval=-1
            )
