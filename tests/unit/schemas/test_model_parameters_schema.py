import pytest
import numpy as np
from schemas.model_parameters_schema import (
    ModelParametersSchema,
    ParameterRangeSchema,
)


class TestParameterRangeSchema:
    """
    ParameterRangeSchemaのユニットテスト
    """

    def test_get_range(self):
        """get_rangeが正しい配列を生成するかテスト"""
        pr = ParameterRangeSchema(min_val=0, max_val=1, step=0.5)
        expected = np.array([0, 0.5])
        np.testing.assert_array_equal(pr.get_range(), expected)

    def test_get_range_empty(self):
        """max_val <= min_val の場合に空の配列を返すかテスト"""
        pr = ParameterRangeSchema(min_val=1, max_val=0, step=0.1)
        assert len(pr.get_range()) == 0

    def test_get_count(self):
        """get_countが正しい数を返すかテスト"""
        pr = ParameterRangeSchema(min_val=10, max_val=20, step=2)
        assert pr.get_count() == 5  # [10, 12, 14, 16, 18]


class TestModelParametersSchema:
    """
    ModelParametersSchemaのユニットテスト
    """

    @pytest.fixture
    def default_params(self):
        """テスト用のデフォルトパラメータを返すフィクスチャ"""
        return {
            "k_min": 1000,
            "k_max": 2000,
            "k_step": 500,
            "gamma_min": 0.1,
            "gamma_max": 0.5,
            "gamma_step": 0.2,
        }

    def test_init(self, default_params):
        """正常に初期化できるかテスト"""
        schema = ModelParametersSchema(**default_params)
        assert isinstance(schema.k_range, ParameterRangeSchema)
        assert isinstance(schema.gamma_range, ParameterRangeSchema)

    def test_get_k_range(self, default_params):
        """get_k_rangeが正しい配列を返すかテスト"""
        schema = ModelParametersSchema(**default_params)
        expected = np.array([1000, 1500])
        np.testing.assert_array_equal(schema.get_k_range(), expected)

    def test_get_gamma_range(self, default_params):
        """get_gamma_rangeが正しい配列を返すかテスト"""
        schema = ModelParametersSchema(**default_params)
        expected = np.array([0.1, 0.3])
        np.testing.assert_allclose(schema.get_gamma_range(), expected)

    def test_get_search_info(self, default_params):
        """get_search_infoが正しい情報を返すかテスト"""
        schema = ModelParametersSchema(**default_params)
        info = schema.get_search_info()
        assert info["k_count"] == 2
        assert info["gamma_count"] == 2
        assert info["total_combinations"] == 4

    def test_update_k_range(self, default_params):
        """update_k_rangeが正しく範囲を更新するかテスト"""
        schema = ModelParametersSchema(**default_params)
        schema.update_k_range(min_val=0, max_val=10, step=5)
        expected = np.array([0, 5])
        np.testing.assert_array_equal(schema.get_k_range(), expected)
        assert schema.get_search_info()["k_count"] == 2

    def test_update_gamma_range(self, default_params):
        """update_gamma_rangeが正しく範囲を更新するかテスト"""
        schema = ModelParametersSchema(**default_params)
        schema.update_gamma_range(min_val=0, max_val=1, step=0.5)
        expected = np.array([0, 0.5])
        np.testing.assert_allclose(schema.get_gamma_range(), expected)
        assert schema.get_search_info()["gamma_count"] == 2
