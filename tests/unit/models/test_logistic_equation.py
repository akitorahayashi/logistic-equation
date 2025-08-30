import pytest
import numpy as np
from models.logistic_equation import LogisticEquationModel


class TestLogisticEquationModel:
    """
    LogisticEquationModelのユニットテスト
    """

    def test_init_valid_parameters(self):
        """正常なパラメータで初期化できることをテスト"""
        model = LogisticEquationModel(gamma=0.1, K=1000)
        assert model.gamma == 0.1
        assert model.K == 1000

    @pytest.mark.parametrize(
        "gamma, K", [(0.1, 0), (0.1, -100), (0.1, np.nan), (0.1, np.inf)]
    )
    def test_differential_equation_invalid_k_raises_error(self, gamma, K):
        """不正なKの値でdifferential_equationを呼ぶとValueErrorが発生することをテスト"""
        model = LogisticEquationModel(gamma=gamma, K=K)
        with pytest.raises(ValueError, match="K は正の有限値である必要があります"):
            model.differential_equation(t=0, v=100)

    def test_differential_equation_v_equals_zero(self):
        """v=0の場合、成長率が0になることをテスト"""
        model = LogisticEquationModel(gamma=0.1, K=1000)
        assert model.differential_equation(t=0, v=0) == 0

    def test_differential_equation_v_equals_k(self):
        """v=Kの場合、成長率が0になることをテスト"""
        model = LogisticEquationModel(gamma=0.1, K=1000)
        assert model.differential_equation(t=0, v=1000) == 0

    def test_differential_equation_v_greater_than_k(self):
        """v > Kの場合、成長率が負になることをテスト"""
        model = LogisticEquationModel(gamma=0.1, K=1000)
        assert model.differential_equation(t=0, v=1100) < 0

    def test_solve_runge_kutta_shape(self):
        """solve_runge_kuttaが出力する配列の形状が正しいことをテスト"""
        model = LogisticEquationModel(gamma=0.03, K=10000)
        t, v = model.solve_runge_kutta(v0=100, t_start=0, t_end=10, dt=0.1)
        assert t.shape == v.shape
        assert len(t) == 101  # (10 - 0) / 0.1 = 100 steps -> 101 points

    @pytest.mark.parametrize(
        "v0, t_start, t_end, dt",
        [
            (100, 10, 10, 0.1),  # t_end == t_start
            (100, 11, 10, 0.1),  # t_end < t_start
            (100, 0, 10, 0),  # dt == 0
            (100, 0, 10, -0.1),  # dt < 0
        ],
    )
    def test_solve_runge_kutta_invalid_time_raises_error(self, v0, t_start, t_end, dt):
        """不正な時間パラメータでsolve_runge_kuttaを呼ぶとValueErrorが発生することをテスト"""
        model = LogisticEquationModel(gamma=0.03, K=10000)
        with pytest.raises(ValueError):
            model.solve_runge_kutta(v0, t_start, t_end, dt)

    def test_solve_runge_kutta_growth(self):
        """v0 < K の場合、値が増加傾向にあることをテスト"""
        model = LogisticEquationModel(gamma=0.03, K=10000)
        t, v = model.solve_runge_kutta(v0=100, t_start=0, t_end=10, dt=1)
        assert v[0] < v[-1]
        assert v[-1] < model.K

    def test_solve_runge_kutta_decay(self):
        """v0 > K の場合、値がKに収束していくことをテスト"""
        model = LogisticEquationModel(gamma=0.03, K=10000)
        t, v = model.solve_runge_kutta(v0=12000, t_start=0, t_end=10, dt=1)
        assert v[0] > v[-1]
        assert v[-1] > model.K

    def test_solve_runge_kutta_handles_overflow(self):
        """Kが非常に小さい場合にエラーを出すことをテスト"""
        # Kが非常に小さいとオーバーフローしやすい
        model = LogisticEquationModel(gamma=1, K=1e-9)
        # The code now raises a ValueError before overflow, which is good.
        # The test should check for this ValueError.
        with pytest.raises(ValueError, match="K は正の有限値である必要があります"):
            model.solve_runge_kutta(v0=1, t_start=0, t_end=10, dt=0.1)
