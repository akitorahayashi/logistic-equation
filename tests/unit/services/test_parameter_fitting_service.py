import pytest
import numpy as np
from unittest.mock import MagicMock, patch
from services.parameter_fitting_service import ParameterFitterService
from models.logistic_equation import LogisticEquationModel
from schemas.model_parameters_schema import ModelParametersSchema


@pytest.fixture
def mock_model_params():
    """ModelParametersSchemaのモックを返すフィクスチャ"""
    mock = MagicMock(spec=ModelParametersSchema)
    mock.get_k_range.return_value = np.array([1000, 2000])
    mock.get_gamma_range.return_value = np.array([0.1, 0.2])
    return mock


@pytest.fixture
def mock_equation_model():
    """LogisticEquationModelのモックを返すフィクスチャ"""
    mock = MagicMock(spec=LogisticEquationModel)
    # (t, v)のタプルを返すように設定
    mock.solve_runge_kutta.return_value = (np.array([0, 1, 2]), np.array([50, 60, 70]))
    return mock


class TestParameterFitterService:
    """
    ParameterFitterServiceのユニットテスト
    """

    def test_init(self, mock_model_params):
        """初期化をテスト"""
        time_data = np.array([0, 1, 2])
        value_data = np.array([50, 65, 75])
        fitter = ParameterFitterService(mock_model_params, time_data, value_data)
        assert fitter.model_params == mock_model_params
        assert np.array_equal(fitter.time_data, time_data)
        assert np.array_equal(fitter.value_data, value_data)
        assert fitter.best_params is None
        assert fitter.min_sse is None

    def test_get_best_params_before_fit_raises_error(self, mock_model_params):
        """fit前にget_best_paramsを呼ぶとエラーになることをテスト"""
        fitter = ParameterFitterService(mock_model_params, np.array([]), np.array([]))
        with pytest.raises(ValueError):
            fitter.get_best_params()

    def test_get_fitted_equation_before_fit_raises_error(self, mock_model_params):
        """fit前にget_fitted_equationを呼ぶとエラーになることをテスト"""
        fitter = ParameterFitterService(mock_model_params, np.array([]), np.array([]))
        with pytest.raises(ValueError):
            fitter.get_fitted_equation()

    @patch("services.parameter_fitting_service.LogisticEquationModel")
    def test_fit_parameters_finds_best_sse(self, MockEquation, mock_model_params):
        """fit_parametersが最小のSSEを見つけられるかテスト"""
        time_data = np.array([0, 1, 2])
        value_data = np.array([50, 60, 80])  # この値に最も近いモデルを探す

        # gamma=0.1, K=1000 -> SSE=100
        # gamma=0.1, K=2000 -> SSE=200
        # gamma=0.2, K=1000 -> SSE=50  <- これが最小
        # gamma=0.2, K=2000 -> SSE=150
        def solve_side_effect(gamma, K):
            if gamma == 0.1 and K == 1000:
                return (time_data, np.array([50, 60, 70]))  # SSE = (80-70)^2 = 100
            if gamma == 0.1 and K == 2000:
                return (
                    time_data,
                    np.array([50, 50, 60]),
                )  # SSE = (60-50)^2+(80-60)^2=500
            if gamma == 0.2 and K == 1000:
                return (
                    time_data,
                    np.array([50, 65, 75]),
                )  # SSE = (60-65)^2+(80-75)^2=50
            if gamma == 0.2 and K == 2000:
                return (
                    time_data,
                    np.array([50, 55, 65]),
                )  # SSE = (60-55)^2+(80-65)^2=250
            return (time_data, np.array([0, 0, 0]))

        # モックのインスタンスがsolve_runge_kuttaを持つように設定
        mock_instance = MagicMock()
        mock_instance.solve_runge_kutta.side_effect = (
            lambda v0, t_start, t_end, dt: solve_side_effect(
                mock_instance.gamma, mock_instance.K
            )
        )

        # クラスが呼ばれたときに、パラメータをインスタンスに設定
        def class_side_effect(gamma, K):
            mock_instance.gamma = gamma
            mock_instance.K = K
            return mock_instance

        MockEquation.side_effect = class_side_effect

        fitter = ParameterFitterService(mock_model_params, time_data, value_data)
        best_params, min_sse = fitter.fit_parameters()

        assert best_params == {"gamma": 0.2, "K": 1000}
        assert pytest.approx(min_sse) == 50
        assert fitter.get_best_params() == best_params
        # fitted_equationのパラメータも更新されていることを確認
        fitted_eq = fitter.get_fitted_equation()
        assert fitted_eq.gamma == 0.2
        assert fitted_eq.K == 1000
