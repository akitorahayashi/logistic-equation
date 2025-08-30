import pytest
import numpy as np
from unittest.mock import MagicMock, patch
from services.visualizer_service import (
    FittingVisualizerService,
    ForecastVisualizerService,
)
from models.logistic_equation import LogisticEquationModel
from schemas.prediction_settings_schema import PredictionSettingsSchema


@pytest.fixture
def mock_settings():
    """PredictionSettingsSchemaのモック"""
    mock = MagicMock(spec=PredictionSettingsSchema)
    mock.start_year = 2000
    return mock


@pytest.fixture
def mock_equation():
    """LogisticEquationModelのモック"""
    mock = MagicMock(spec=LogisticEquationModel)
    mock.gamma = 0.1
    mock.K = 10000  # 1万
    # v0<Kなので成長するデータ
    mock.solve_runge_kutta.return_value = (np.arange(10), np.linspace(100, 500, 10))
    return mock


@patch("services.visualizer_service.plt")
class TestFittingVisualizerService:
    """
    FittingVisualizerServiceのユニットテスト
    """

    def test_plot_with_equation_calls_plot(
        self, mock_plt, mock_settings, mock_equation
    ):
        """plot_with_equationがmatplotlibの関数を呼び出すことをテスト"""
        visualizer = FittingVisualizerService(mock_settings)
        time_array = np.arange(10)
        value_array = np.linspace(100, 450, 10)

        visualizer.plot_with_equation(time_array, value_array, mock_equation)

        mock_plt.figure.assert_called()
        ax = mock_plt.figure.return_value.add_subplot.return_value
        # 2回プロットが呼ばれるか (実データとモデル)
        assert ax.plot.call_count == 2
        ax.set_title.assert_called()
        ax.set_xlabel.assert_called()
        ax.set_ylabel.assert_called()
        ax.legend.assert_called()
        ax.grid.assert_called_with(True)

    def test_scaling_logic(self, mock_plt, mock_settings, mock_equation):
        """値のスケールが正しく行われるかテスト"""
        visualizer = FittingVisualizerService(mock_settings)
        # K=1億なので、「億」単位にスケールされるはず
        mock_equation.K = 10**8
        time_array = np.arange(2)
        value_array = np.array([10**7, 1.1 * 10**7])

        # モデルの返す値もスケールに合わせる
        mock_equation.solve_runge_kutta.return_value = (
            time_array,
            np.array([10**7, 1.2 * 10**7]),
        )

        visualizer.plot_with_equation(time_array, value_array, mock_equation)
        ax = mock_plt.figure.return_value.add_subplot.return_value

        # ylabelが「億」になっているか
        ax.set_ylabel.assert_called_with("値 (億)")

        # plotに渡される値がスケールされているか
        # 最初のplot呼び出し(実データ)の2番目の引数(y値)
        scaled_values_arg = ax.plot.call_args_list[0].args[1]
        expected_scaled_values = value_array / 10**8
        np.testing.assert_allclose(scaled_values_arg, expected_scaled_values)


@patch("services.visualizer_service.plt")
class TestForecastVisualizerService:
    """
    ForecastVisualizerServiceのユニットテスト
    """

    def test_plot_forecast_calls_plot(self, mock_plt, mock_settings):
        """plot_forecastがmatplotlibの関数を呼び出すことをテスト"""
        visualizer = ForecastVisualizerService(mock_settings)
        time_array = np.arange(5)
        value_array = np.linspace(10, 50, 5)
        forecast_time = np.arange(5, 10)
        forecast_value = np.linspace(50, 80, 5)

        visualizer.plot_forecast(time_array, value_array, forecast_time, forecast_value)

        mock_plt.figure.assert_called()
        ax = mock_plt.figure.return_value.add_subplot.return_value
        assert ax.plot.call_count == 2  # 実績と予測
        ax.axvline.assert_called()  # 予測開始線
        ax.set_title.assert_called()
        ax.legend.assert_called()
        ax.grid.assert_called_with(True)
